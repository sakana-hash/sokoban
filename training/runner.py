"""Collect batched experience on CPU and update the Q network on the selected device."""

import json
import random
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch

from .curriculum import next_stage, single_box_easy_probability, validation_games
from .dqn import ReplayBuffer, make_network, mask_batch, optimize, state_batch
from .env import Game, Point, encode, legal_mask, make_training_game, potential, shaped_reward, step_game
from .evaluate import play_batch
from .runtime import checkpoint_version


DISCOUNT = 0.99


class Episode:
    def __init__(
        self, game: Game, version: int, length: int = 0, visited_since_push: Iterable[Point] = (),
    ):
        self.game = game
        self.length = length
        self.state = encode(game)
        self.mask = legal_mask(game)
        self.potential = potential(game) if version == 2 else 0.0
        self.visited_since_push = {game.player, *visited_since_push} if version == 2 else None

    def snapshot(self) -> dict:
        snapshot = {"board": self.game.observation()["board"], "length": self.length}
        if self.visited_since_push is not None:
            snapshot["visited_since_push"] = sorted(self.visited_since_push)
        return snapshot


class Trainer:
    def __init__(self, config, device, checkpoint):
        self.config = config
        self.device = device
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.version = 2 if checkpoint is None else checkpoint_version(checkpoint)
        self.online = make_network(self.version).to(device)
        self.target = make_network(self.version).to(device).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=3e-4)
        self.replay = ReplayBuffer(config.replay_size)
        self.steps = 0
        self.updates = 0
        self.episodes = 0
        self.episodes_won = 0
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
        extra_pulls = None
        if self.version == 1:
            self.stage = min(4, 1 + self.steps // self.config.curriculum_steps)
        elif self.stage == 1 and self.rng.random() < single_box_easy_probability(self.steps):
            extra_pulls = 1
        game = make_training_game(self.rng, self.stage, max_extra_pulls=extra_pulls)
        return Episode(game, self.version)

    def epsilon(self):
        elapsed = max(0, self.steps - self.config.warmup)
        return 1.0 - 0.95 * min(1.0, elapsed / self.config.epsilon_steps)

    @torch.no_grad()
    def greedy(self, states, masks):
        values = self.online(state_batch(states, self.device))
        legal = mask_batch(masks, self.device)
        return values.masked_fill(~legal, -torch.inf).argmax(dim=1).tolist()

    def actions(self, slots):
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
            greedy = self.greedy([slots[i].state for i in exploit], [slots[i].mask for i in exploit])
            for i, action in zip(exploit, greedy):
                actions[i] = action
        return actions

    def advance(self, count):
        slots = self.slots[:count]
        before = self.steps
        for i, (slot, action) in enumerate(zip(slots, self.actions(slots))):
            previous = slot.state
            reward = step_game(slot.game, action, slot.visited_since_push)
            slot.length += 1
            slot.state = encode(slot.game)
            slot.mask = legal_mask(slot.game)
            won = slot.game.won
            terminal = won or not any(slot.mask)
            if self.version == 2:
                after = potential(slot.game)
                reward = shaped_reward(reward, slot.potential, after, terminal, DISCOUNT)
                slot.potential = after
            self.replay.add(previous, action, reward, slot.state, slot.mask, terminal)
            self.steps += 1
            if terminal or slot.length >= self.config.max_episode_steps:
                # A time limit resets the collector, but the stored state can still bootstrap.
                self.episodes += 1
                self.episodes_won += int(won)
                self.recent_wins.append(won)
                self.slots[i] = self.new_episode()

        first_update = max(before + 1, self.config.warmup, self.config.batch_size)
        for _ in range((first_update + 3) // 4 * 4, self.steps + 1, 4):
            self.last_loss = optimize(
                self.online, self.target, self.optimizer, self.replay,
                self.config.batch_size, DISCOUNT, self.device, self.rng,
            )
            self.updates += 1
            if self.updates % 500 == 0:
                self.target.load_state_dict(self.online.state_dict())

    def policy(self):
        return {"version": self.version, "steps": self.steps, "online": self.online.state_dict()}

    def validate(self):
        per_count = self.config.validation_rooms
        wins = [0] * 4
        games = [game for _, game in validation_games(per_count)]
        for start in range(0, len(games), self.config.batch_size):
            play_batch(games[start:start + self.config.batch_size], self.greedy, 256)
        for game in games:
            wins[len(game.boxes) - 1] += int(game.won)
        score = sum(wins)
        if self.best_policy is None or score > self.best_policy["wins"]:
            self.best_policy = self.policy() | {"wins": score}
            self.best_policy["online"] = {
                key: value.detach().to("cpu", copy=True) for key, value in self.online.state_dict().items()
            }
        self.stage = next_stage(self.stage, wins, per_count)
        return {
            "validation_steps": self.steps, "wins_by_boxes_1_to_4": wins, "rooms_per_group": per_count,
            "training_max_boxes": self.stage, "best_validation_step": self.best_policy["steps"],
        }

    def loss_value(self):
        return float(self.last_loss) if self.last_loss is not None else None

    def episode_counts(self):
        return {
            "episodes": self.episodes,
            "episodes_won": self.episodes_won,
            "episodes_lost": self.episodes - self.episodes_won - self.episodes_unknown,
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
        self.replay.load_state_dict(checkpoint["replay"])
        self.steps = checkpoint["steps"]
        self.updates = checkpoint["updates"]
        self.episodes = checkpoint["episodes"]
        self.last_loss = checkpoint["last_loss"]
        self.recent_wins.extend(checkpoint["recent_wins"])
        # Older checkpoints only kept outcomes for the last 100 completed episodes.
        self.episodes_won = checkpoint.get("episodes_won", sum(self.recent_wins))
        self.episodes_unknown = checkpoint.get("episodes_unknown", self.episodes - len(self.recent_wins))
        self.stage = checkpoint.get("stage", min(4, 1 + self.steps // self.config.curriculum_steps))
        self.best_policy = checkpoint.get("best_policy")
        self.slots = [
            Episode(Game(s["board"]), self.version, s["length"], s.get("visited_since_push", ()))
            for s in checkpoint["slots"]
        ]
        self.rng.setstate(checkpoint["rng"])
        torch.set_rng_state(checkpoint["torch_rng"])

    def training_metrics(self):
        rate = sum(self.recent_wins) / len(self.recent_wins) if self.recent_wins else None
        loss = self.loss_value()
        then, old_steps, old_updates = self.last_log
        now = perf_counter()
        self.last_log = (now, self.steps, self.updates)
        return {
            "steps": self.steps, "updates": self.updates, **self.episode_counts(),
            "epsilon": round(self.epsilon(), 4), "loss": loss, "training_max_boxes": self.stage,
            "full_start_probability": round(1 - single_box_easy_probability(self.steps), 4)
            if self.version == 2 and self.stage == 1 else 1.0,
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


def run(config, device, goal, save_every, output, checkpoint, resume):
    # A fresh run owns a new directory. A resumed run may continue in its own directory.
    continuing = resume is not None and output.resolve() == resume.resolve().parent
    output.mkdir(parents=True, exist_ok=continuing)
    trainer = Trainer(config, device, checkpoint)
    if checkpoint is not None:
        # Drop the loaded replay snapshot so evicted samples can be reclaimed.
        checkpoint.clear()
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as history:
        def record(event: str, values: dict, message: str | None = None):
            entry = {"event": event, "timestamp": datetime.now(UTC).isoformat(), **values}
            history.write(json.dumps(entry, ensure_ascii=False) + "\n")
            history.flush()
            print(message if message is not None else json.dumps(values), flush=True)

        record("start", {
            "device": str(device), "version": trainer.version, "steps": trainer.steps,
            "target_steps": goal, "save_every": save_every, "config": asdict(config),
            "resume": str(resume) if resume is not None else None, "output": str(output),
        }, f"device={device}  version={trainer.version}  steps={trainer.steps}→{goal}  output={output}")
        while trainer.steps < goal:
            before = trainer.steps
            trainer.advance(min(config.num_envs, goal - before))
            if trainer.steps // 2000 != before // 2000 or trainer.steps == goal:
                record("train", trainer.training_metrics())
            crossed_validation_interval = trainer.steps // config.curriculum_steps != before // config.curriculum_steps
            validation_due = trainer.version == 2 and crossed_validation_interval
            if validation_due:
                record("validation", trainer.validate())
            if validation_due or trainer.steps // save_every != before // save_every or trainer.steps == goal:
                trainer.save(output)
