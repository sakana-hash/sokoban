"""Collect experience, update the Q networks, and persist their training state."""

import random
from collections import deque
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch

from ..curriculum import (
    make_single_box_game, next_stage, single_box_easy_probability, single_box_small_board_probability,
    validation_games,
)
from ..dqn import ReplayBuffer, make_network, mask_batch, optimize, state_batch
from ..env import Game, is_deadlocked, make_training_game
from ..evaluate import play_batch
from ..profiling import null_region
from ..runtime import checkpoint_version
from .episode import DISCOUNT, Episode


class Trainer:
    def __init__(self, config, device, checkpoint, *, channels_last=False):
        self.config = config
        self.device = device
        self.channels_last = channels_last
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.version = config.model_version if checkpoint is None else checkpoint_version(checkpoint)
        layout = torch.channels_last if channels_last else torch.contiguous_format
        self.online = make_network(self.version).to(device, memory_format=layout)
        self.target = make_network(self.version).to(device, memory_format=layout).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=3e-4)
        self.replay = ReplayBuffer(config.replay_size)
        self.steps = 0
        self.updates = 0
        self.episodes = 0
        self.episodes_won = 0
        self.episodes_deadlocked = 0
        self.episodes_unknown = 0
        self.recent_wins = deque(maxlen=100)
        self.last_loss = None
        self.stage = 1
        self.best_policy = None

        if checkpoint is None:
            self.target.load_state_dict(self.online.state_dict())
            self.slots = [self.new_episode() for _ in range(config.num_envs)]
        else:
            self.restore(checkpoint)
        self.last_log = (perf_counter(), self.steps, self.updates)

    def new_episode(self):
        if self.version == 1:
            self.stage = min(4, 1 + self.steps // self.config.curriculum_steps)
        if self.version >= 2 and self.stage == 1:
            game = make_single_box_game(
                self.rng, self.steps, small_board_curriculum=self.config.small_board_curriculum,
                anneal_easy_starts=self.config.anneal_easy_starts,
            )
        else:
            game = make_training_game(self.rng, self.stage)
        return Episode(game, self.version)

    def epsilon(self):
        elapsed = max(0, self.steps - self.config.warmup)
        return 1.0 - 0.95 * min(1.0, elapsed / self.config.epsilon_steps)

    def finish_episode(self, *, won: bool, deadlocked: bool) -> Episode:
        self.episodes += 1
        self.episodes_won += int(won)
        self.episodes_deadlocked += int(deadlocked)
        self.recent_wins.append(won)
        return self.new_episode()

    @torch.no_grad()
    def greedy(self, states, masks, *, region=null_region):
        boards = state_batch(states, self.device, channels_last=self.channels_last, region=region)
        with region("policy_compute"):
            values = self.online(boards)
        legal = mask_batch(masks, self.device, region=region)
        with region("policy_compute"):
            actions = values.masked_fill(~legal, -torch.inf).argmax(dim=1)
        with region("transfer"):
            return actions.tolist()

    def actions(self, slots, *, region=null_region):
        with region("action_selection"):
            epsilon = self.epsilon()
            actions = []
            exploit = []
            for i, slot in enumerate(slots):
                if self.rng.random() < epsilon:
                    actions.append(self.rng.choice([a for a, legal in enumerate(slot.mask) if legal]))
                else:
                    actions.append(None)
                    exploit.append(i)
        if exploit:
            greedy = self.greedy([slots[i].state for i in exploit], [slots[i].mask for i in exploit], region=region)
            for i, action in zip(exploit, greedy):
                actions[i] = action
        return actions

    def advance_environments(self, slots, actions):
        for i, (slot, action) in enumerate(zip(slots, actions, strict=True)):
            previous = slot.state
            reward, terminal, deadlocked = slot.step(action, self.config)
            self.replay.add(previous, action, reward, slot.state, slot.mask, terminal)
            self.steps += 1
            if terminal or slot.length >= self.config.max_episode_steps:
                # A time limit resets the collector, but the stored state can still bootstrap.
                self.slots[i] = self.finish_episode(won=slot.game.won, deadlocked=deadlocked)

    def advance(self, count, *, region=null_region):
        slots = self.slots[:count]
        before = self.steps
        actions = self.actions(slots, region=region)
        with region("environment"):
            self.advance_environments(slots, actions)
        first_update = max(before + 1, self.config.warmup, self.config.batch_size)
        for _ in range((first_update + 3) // 4 * 4, self.steps + 1, 4):
            self.last_loss = optimize(
                self.online, self.target, self.optimizer, self.replay,
                self.config.batch_size, DISCOUNT, self.device, self.rng,
                channels_last=self.channels_last, region=region,
            )
            self.updates += 1
            if self.updates % 500 == 0:
                with region("train_compute"):
                    self.target.load_state_dict(self.online.state_dict())

    def policy(self):
        return {"version": self.version, "steps": self.steps, "online": self.online.state_dict()}

    def validate(self):
        per_count = self.config.validation_rooms
        wins = [0] * 4
        deadlocks = [0] * 4
        games = [game for _, game in validation_games(per_count)]
        for start in range(0, len(games), self.config.batch_size):
            batch = games[start:start + self.config.batch_size]
            outcomes = play_batch(batch, self.greedy, 256, deadlock_detection=self.config.deadlock_detection)
            for game, outcome in zip(batch, outcomes, strict=True):
                group = len(game.boxes) - 1
                wins[group] += int(game.won)
                deadlocks[group] += int(outcome == "死锁")
        score = sum(wins)
        if self.best_policy is None or score > self.best_policy["wins"]:
            self.best_policy = self.policy() | {"wins": score}
            self.best_policy["online"] = {
                key: value.detach().to("cpu", copy=True) for key, value in self.online.state_dict().items()
            }
        self.stage = next_stage(self.stage, wins, per_count)
        return {
            "validation_steps": self.steps, "wins_by_boxes_1_to_4": wins, "rooms_per_group": per_count,
            "deadlocks_by_boxes_1_to_4": deadlocks,
            "training_max_boxes": self.stage, "best_validation_step": self.best_policy["steps"],
        }

    def loss_value(self):
        return float(self.last_loss) if self.last_loss is not None else None

    def episode_counts(self):
        return {
            "episodes": self.episodes,
            "episodes_won": self.episodes_won,
            "episodes_lost": self.episodes - self.episodes_won - self.episodes_unknown,
            "episodes_deadlocked": self.episodes_deadlocked,
            "episodes_unknown": self.episodes_unknown,
        }

    def save(self, output):
        policy = self.policy()
        full = {
            **policy,
            "config": asdict(self.config),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "updates": self.updates,
            **self.episode_counts(),
            "recent_wins": list(self.recent_wins),
            "last_loss": self.loss_value(),
            "stage": self.stage,
            "best_policy": self.best_policy,
            "slots": [slot.snapshot() for slot in self.slots],
            "rng": self.rng.getstate(),
            "torch_rng": torch.get_rng_state(),
        }
        atomic_save(full, output / "latest.pt")
        atomic_save(policy, output / "policy.pt")
        if self.best_policy:
            atomic_save(self.best_policy, output / "best.pt")

    def restore(self, checkpoint):
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Adam restores moment tensors with their saved strides when switching layouts.
        layout = torch.channels_last if self.channels_last else torch.contiguous_format
        for state in self.optimizer.state.values():
            for name, tensor in state.items():
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                    state[name] = tensor.to(memory_format=layout)
        self.replay.load_state_dict(checkpoint["replay"])
        self.steps = checkpoint["steps"]
        self.updates = checkpoint["updates"]
        self.episodes = checkpoint["episodes"]
        self.last_loss = checkpoint["last_loss"]
        self.recent_wins.extend(checkpoint["recent_wins"])
        # Older checkpoints only kept outcomes for the last 100 completed episodes.
        self.episodes_won = checkpoint.get("episodes_won", sum(self.recent_wins))
        self.episodes_deadlocked = checkpoint.get("episodes_deadlocked", 0)
        self.episodes_unknown = checkpoint.get("episodes_unknown", self.episodes - len(self.recent_wins))
        self.stage = checkpoint.get("stage", min(4, 1 + self.steps // self.config.curriculum_steps))
        self.best_policy = checkpoint.get("best_policy")
        self.slots = [
            Episode(Game(s["board"]), self.version, s["length"], s.get("visited_since_push", ()))
            for s in checkpoint["slots"]
        ]
        self.rng.setstate(checkpoint["rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        # A checkpoint from before deadlock detection can still contain a stranded box.
        for i, slot in enumerate(self.slots):
            if self.config.deadlock_detection and is_deadlocked(slot.game):
                self.slots[i] = self.finish_episode(won=False, deadlocked=True)

    def training_metrics(self):
        rate = sum(self.recent_wins) / len(self.recent_wins) if self.recent_wins else None
        loss = self.loss_value()
        then, old_steps, old_updates = self.last_log
        now = perf_counter()
        self.last_log = (now, self.steps, self.updates)
        single_box_stage = self.version >= 2 and self.stage == 1
        easy_probability = single_box_easy_probability(self.steps, anneal=self.config.anneal_easy_starts)
        return {
            "steps": self.steps, "updates": self.updates, **self.episode_counts(),
            "epsilon": round(self.epsilon(), 4), "loss": loss, "training_max_boxes": self.stage,
            "full_start_probability": round(1 - easy_probability, 4) if single_box_stage else 1.0,
            "small_board_probability": round(single_box_small_board_probability(self.steps), 4)
            if single_box_stage and self.config.small_board_curriculum else 0.0,
            "decisions_per_sec": round((self.steps - old_steps) / (now - then), 1),
            "updates_per_sec": round((self.updates - old_updates) / (now - then), 1),
            "win_rate_last_100": rate,
        }


def atomic_save(payload, path):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
