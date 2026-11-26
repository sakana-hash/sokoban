"""Schedule environment batches, validation, profiling and checkpoint writes."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..profiling import TrainingProfile
from .trainer import Trainer


def profile_advances(trainer, goal, save_every, output, max_rounds):
    start = trainer.steps
    initial_updates = trainer.updates
    initial_epsilon = trainer.epsilon()
    # Finish the last regular collector batch before doing any logging, validation or I/O.
    intervals = [2000, save_every]
    if trainer.version >= 2:
        intervals.append(trainer.config.curriculum_steps)
    next_boundary = min((start // n + 1) * n for n in intervals)
    profile = TrainingProfile(trainer.device)
    profile.start()
    rounds = 0
    try:
        while rounds < max_rounds and trainer.steps < min(goal, next_boundary):
            trainer.advance(min(trainer.config.num_envs, goal - trainer.steps), region=profile.region)
            rounds += 1
    except BaseException:
        profile.abort()
        raise
    return profile.stop(output, start_step=start, end_step=trainer.steps, rounds=rounds,
                        updates=trainer.updates - initial_updates,
                        epsilon_start=initial_epsilon, epsilon_end=trainer.epsilon())


def run(config, device, goal, save_every, output, checkpoint, resume, *,
        profile_start=None, profile_rounds=16, channels_last=False) -> Path:
    # A fresh run owns a new directory. A resumed run may continue in its own directory.
    continuing = resume is not None and output.resolve() == resume.resolve().parent
    output.mkdir(parents=True, exist_ok=continuing)
    trainer = Trainer(config, device, checkpoint, channels_last=channels_last)
    if checkpoint is not None:
        # Drop the loaded replay snapshot so evicted samples can be reclaimed.
        checkpoint.clear()
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as history:
        def record(event: str, values: dict, message: str | None = None):
            entry = {"event": event, "timestamp": datetime.now(UTC).isoformat(), **values}
            history.write(json.dumps(entry, ensure_ascii=False) + "\n")
            history.flush()
            print(message if message is not None else json.dumps(values), flush=True)

        parameters = sum(p.numel() for p in trainer.online.parameters() if p.requires_grad)
        record("start", {
            "device": str(device), "version": trainer.version, "steps": trainer.steps,
            "trainable_parameters": parameters, "channels_last": channels_last,
            "target_steps": goal, "save_every": save_every, "config": asdict(config),
            "resume": str(resume) if resume is not None else None, "output": str(output),
        }, (f"device={device}  version={trainer.version}  parameters={parameters}  "
            f"steps={trainer.steps}→{goal}  channels_last={channels_last}  output={output}"))
        while trainer.steps < goal:
            before = trainer.steps
            count = min(config.num_envs, goal - before)
            if profile_start is not None and before + count > profile_start:
                summary = profile_advances(trainer, goal, save_every, output, profile_rounds)
                record("profile", {"start_step": before, "end_step": trainer.steps, "summary": str(summary)})
                profile_start = None
            else:
                trainer.advance(count)
            if trainer.steps // 2000 != before // 2000 or trainer.steps == goal:
                record("train", trainer.training_metrics())
            crossed_validation_interval = trainer.steps // config.curriculum_steps != before // config.curriculum_steps
            validation_due = trainer.version >= 2 and crossed_validation_interval
            if validation_due:
                record("validation", trainer.validate())
            if validation_due or trainer.steps // save_every != before // save_every or trainer.steps == goal:
                trainer.save(output)
    return output / ("best.pt" if trainer.best_policy is not None else "policy.pt")
