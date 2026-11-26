"""Collect batched experience on CPU and update the Q network on the selected device."""

import json
import random
from collections import deque
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .dqn import QNetwork, ReplayBuffer, optimize, state_batch
from .env import Game, encode, legal_mask, make_training_game, step_game


class Episode:
    def __init__(self, game: Game, length: int = 0):
        self.game = game
        self.length = length
        self.state = encode(game)
        self.mask = legal_mask(game)

    def snapshot(self) -> dict:
        return {"board": self.game.observation()["board"], "length": self.length}


class Trainer:
    def __init__(self, config, device, checkpoint):
        self.config = config
        self.device = device
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.online = QNetwork().to(device)
        self.target = QNetwork().to(device).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=3e-4)
        self.replay = ReplayBuffer(config.replay_size)
        self.steps = 0
        self.updates = 0
        self.episodes = 0
        self.recent_wins = deque(maxlen=100)
        self.last_loss = None

        if checkpoint is None:
            self.target.load_state_dict(self.online.state_dict())
            self.slots = [self.new_episode() for _ in range(config.num_envs)]
        else:
            self.restore(checkpoint)

    def new_episode(self):
        max_boxes = min(4, 1 + self.steps // self.config.curriculum_steps)
        return Episode(make_training_game(self.rng, max_boxes))

    def epsilon(self):
        elapsed = max(0, self.steps - self.config.warmup)
        return 1.0 - 0.95 * min(1.0, elapsed / self.config.epsilon_steps)

    @torch.no_grad()
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
            states = state_batch([slots[i].state for i in exploit], self.device)
            mask = torch.tensor([slots[i].mask for i in exploit], device=self.device, dtype=torch.bool)
            greedy = self.online(states).masked_fill(~mask, -torch.inf).argmax(dim=1).tolist()
            for i, action in zip(exploit, greedy):
                actions[i] = action
        return actions

    def advance(self, count):
        slots = self.slots[:count]
        before = self.steps
        for i, (slot, action) in enumerate(zip(slots, self.actions(slots))):
            previous = slot.state
            reward = step_game(slot.game, action)
            slot.length += 1
            slot.state = encode(slot.game)
            slot.mask = legal_mask(slot.game)
            terminal = slot.game.won or not any(slot.mask)
            self.replay.add(previous, action, reward, slot.state, slot.mask, terminal)
            self.steps += 1
            if terminal or slot.length >= self.config.max_episode_steps:
                # A time limit resets the collector, but the stored state can still bootstrap.
                self.episodes += 1
                self.recent_wins.append(slot.game.won)
                self.slots[i] = self.new_episode()

        first_update = max(before + 1, self.config.warmup, self.config.batch_size)
        for _ in range((first_update + 3) // 4 * 4, self.steps + 1, 4):
            self.last_loss = optimize(
                self.online, self.target, self.optimizer, self.replay,
                self.config.batch_size, 0.99, self.device, self.rng,
            )
            self.updates += 1
            if self.updates % 500 == 0:
                self.target.load_state_dict(self.online.state_dict())

    def save(self, output):
        policy = {"version": 1, "steps": self.steps, "online": self.online.state_dict()}
        full = {
            **policy,
            "config": asdict(self.config),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "updates": self.updates,
            "episodes": self.episodes,
            "recent_wins": list(self.recent_wins),
            "last_loss": self.last_loss,
            "slots": [slot.snapshot() for slot in self.slots],
            "rng": self.rng.getstate(),
            "torch_rng": torch.get_rng_state(),
        }
        atomic_save(full, output / "latest.pt")
        atomic_save(policy, output / "policy.pt")

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
        self.slots = [Episode(Game(s["board"]), s["length"]) for s in checkpoint["slots"]]
        self.rng.setstate(checkpoint["rng"])
        torch.set_rng_state(checkpoint["torch_rng"])

    def log(self):
        rate = sum(self.recent_wins) / len(self.recent_wins) if self.recent_wins else None
        print(json.dumps({
            "steps": self.steps, "updates": self.updates, "episodes": self.episodes,
            "epsilon": round(self.epsilon(), 4), "loss": self.last_loss,
            "win_rate_last_100": rate,
        }), flush=True)


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
    print(f"device={device}  steps={trainer.steps}→{goal}  output={output}", flush=True)
    while trainer.steps < goal:
        before = trainer.steps
        trainer.advance(min(config.num_envs, goal - before))
        if trainer.steps // 2000 != before // 2000 or trainer.steps == goal:
            trainer.log()
        if trainer.steps // save_every != before // save_every or trainer.steps == goal:
            trainer.save(output)
