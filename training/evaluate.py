"""Benchmark a frozen Double DQN and a matched random policy on reproducible boards."""

import argparse
import json
import random
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from .env import encode, legal_mask, make_training_game, step_game
from sokoban import Game, LEVELS


def play_batch(
    games: Sequence[Game],
    choose_batch: Callable[[Sequence[bytes], Sequence[Sequence[int]]], Sequence[int]],
    max_steps: int,
    *,
    detect_cycle: bool = True,
) -> list[str]:
    """Make one inference call for all games awaiting a decision in each round."""
    limits = [g.moves + max_steps for g in games]
    seen = [set() for _ in games]
    outcomes = [""] * len(games)
    active = list(range(len(games)))
    while active:
        states = []
        masks = []
        pending = []
        for i in active:
            game = games[i]
            if game.won or game.moves >= limits[i]:
                outcomes[i] = "通关" if game.won else "达到步数上限"
                continue
            state = encode(game)
            if detect_cycle and state in seen[i]:
                outcomes[i] = "循环"
                continue
            mask = legal_mask(game)
            if not any(mask):
                outcomes[i] = "无可用动作"
                continue
            if detect_cycle:
                seen[i].add(state)
            states.append(state)
            masks.append(mask)
            pending.append(i)

        if not pending:
            break
        actions = choose_batch(states, masks)
        for i, action in zip(pending, actions, strict=True):
            step_game(games[i], action)
        active = pending
    return outcomes


def play_level(
    game: Game,
    choose_action: Callable[[bytes, Sequence[int]], int],
    max_steps: int,
    *,
    detect_cycle: bool = True,
) -> str:
    """Play up to max_steps primitive moves. Disable cycle detection for stochastic policies."""
    def choose_one(states, masks):
        return [choose_action(states[0], masks[0])]

    return play_batch([game], choose_one, max_steps, detect_cycle=detect_cycle)[0]


def evaluation_boards(
    generated: int | None, max_boxes: int, seed: int
) -> Iterator[tuple[str, Sequence[str]]]:
    if generated is None:
        yield from LEVELS
    else:
        maps = random.Random(seed)
        for index in range(1, generated + 1):
            game = make_training_game(maps, max_boxes)
            yield f"生成关 {index}", game.observation()["board"]


def save_report(checkpoint: Path, report: dict) -> Path:
    directory = checkpoint.parent / "evaluations"
    directory.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC)
    prefix = f"{timestamp:%Y%m%dT%H%M%S%fZ}-{checkpoint.stem}-"
    with NamedTemporaryFile("w", encoding="utf-8", dir=directory, prefix=prefix, suffix=".tmp") as stream:
        json.dump({"timestamp": timestamp.isoformat(), **report}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        # Publish the complete file atomically, without replacing another report.
        path = Path(stream.name).with_suffix(".json")
        path.hardlink_to(stream.name)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Double DQN 在内置关卡或固定随机地图上的表现")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练保存的检查点")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-steps", type=int, default=256, help="每关角色移动步数上限，含走路和推箱")
    parser.add_argument("--generated", type=int, metavar="N", help="评估 N 个生成关卡，默认内置 30 关")
    parser.add_argument("--max-boxes", type=int, choices=range(1, 5), default=4, help="生成地图的箱数上限")
    parser.add_argument("--seed", type=int, default=2026, help="固定生成地图与随机基线的种子")
    parser.add_argument("--compare-random", action="store_true", help="在相同开局对照均匀随机合法动作")
    args = parser.parse_args()
    if args.max_steps < 1 or (args.generated is not None and args.generated < 1):
        parser.error("步数上限和生成地图数量必须大于 0")
    if not args.checkpoint.is_file():
        parser.error(f"找不到检查点 {args.checkpoint}")

    try:
        import torch
    except ModuleNotFoundError:
        parser.exit(1, "请在已安装训练依赖的环境中运行评估。\n")
    from .dqn import make_network, mask_batch, state_batch
    from .runtime import checkpoint_version, select_device

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        version = checkpoint_version(checkpoint)
        device = select_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    if "online" not in checkpoint:
        parser.error("检查点缺少在线网络权重 online")
    model = make_network(version)
    model.load_state_dict(checkpoint["online"])
    checkpoint_steps = checkpoint.get("steps")
    del checkpoint
    model.to(device).eval()

    def greedy(state: bytes, mask: Sequence[int]) -> int:
        values = model(state_batch([state], device))
        allowed = mask_batch([mask], device)
        return int(values.masked_fill(~allowed, -torch.inf).argmax(dim=1).item())

    with torch.inference_mode():
        run_evaluation(args, greedy, str(device), version, checkpoint_steps)


def run_evaluation(
    args: argparse.Namespace, greedy: Callable[[bytes, Sequence[int]], int],
    device: str, version: int, checkpoint_steps: int | None,
) -> None:
    # This stream never advances the map generator. Both policies see identical maps.
    actions_rng = random.Random(args.seed + 1)

    def random_action(state: bytes, mask: Sequence[int]) -> int:
        return actions_rng.choice([action for action, legal in enumerate(mask) if legal])

    attempts = Counter()
    wins = Counter()
    random_wins = Counter()
    outcomes = Counter()
    zero_push = 0
    levels = []
    source = "内置关卡" if args.generated is None else f"生成 {args.generated} 关，最多 {args.max_boxes} 箱"
    print(f"{source}，种子 {args.seed}，设备 {device}，模型 v{version}，每关最多移动 {args.max_steps} 步")
    boards = evaluation_boards(args.generated, args.max_boxes, args.seed)
    for level, (title, rows) in enumerate(boards, start=1):
        game = Game(rows)
        boxes = len(game.boxes)
        outcome = play_level(game, greedy, args.max_steps)
        attempts[boxes] += 1
        wins[boxes] += int(game.won)
        outcomes[outcome] += 1
        zero_push += int(game.pushes == 0)
        record = {
            "level": level, "title": title, "boxes": boxes,
            "model": {"outcome": outcome, "moves": game.moves, "pushes": game.pushes},
        }
        if args.generated is None:
            print(f"{level:02d} {title} | {boxes} 箱 | {outcome} | {game.moves} 步 {game.pushes} 推")
        if args.compare_random:
            baseline = Game(rows)
            random_outcome = play_level(baseline, random_action, args.max_steps, detect_cycle=False)
            random_wins[boxes] += int(baseline.won)
            record["random"] = {"outcome": random_outcome, "moves": baseline.moves, "pushes": baseline.pushes}
        levels.append(record)

    def report(label: str, won: int, trials: int, random_won: int) -> dict:
        counts = {"wins": won, "attempts": trials}
        result = f"{label}通关 {won}/{trials}（{won / trials:.1%}）"
        if args.compare_random:
            result += f" | 随机基线 {random_won}/{trials}（{random_won / trials:.1%}）"
            counts["random_wins"] = random_won
        print(result)
        return counts

    by_boxes = {}
    for boxes, count in sorted(attempts.items()):
        by_boxes[boxes] = report(f"{boxes} 箱", wins[boxes], count, random_wins[boxes])
    summary = report("总", wins.total(), attempts.total(), random_wins.total())
    summary.update(cycles=outcomes["循环"], timeouts=outcomes["达到步数上限"],
                   no_legal_actions=outcomes["无可用动作"], zero_push=zero_push)
    print(f"模型循环 {outcomes['循环']} 局，步数耗尽 {outcomes['达到步数上限']} 局，"
          f"无可用动作 {outcomes['无可用动作']} 局，零次推动 {zero_push} 局")
    path = save_report(args.checkpoint, {
        "checkpoint": args.checkpoint.name,
        "checkpoint_steps": checkpoint_steps, "version": version, "device": device,
        "source": "builtin" if args.generated is None else "generated",
        "seed": args.seed, "generated": args.generated,
        "max_boxes": args.max_boxes if args.generated is not None else None,
        "max_steps": args.max_steps, "compare_random": args.compare_random,
        "summary": summary, "by_boxes": by_boxes, "levels": levels,
    })
    print(f"评估报告已保存到 {path}")


if __name__ == "__main__":
    main()
