"""Benchmark a frozen Double DQN and a matched random policy on reproducible boards."""

import argparse
import json
import random
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from .curriculum import SMALL_BOARD_MAX_SIDE
from .env import encode, is_deadlocked, legal_mask, make_training_game, pushed_into_deadlock, step_game
from sokoban import Game, LEVELS


def play_batch(
    games: Sequence[Game],
    choose_batch: Callable[[Sequence[bytes], Sequence[Sequence[int]]], Sequence[int]],
    max_steps: int,
    *,
    detect_cycle: bool = True,
    deadlock_detection: bool = True,
) -> list[str]:
    """Make one inference call for all games awaiting a decision in each round."""
    limits = [g.moves + max_steps for g in games]
    seen = [set() for _ in games]
    outcomes = ["死锁" if deadlock_detection and is_deadlocked(game) else "" for game in games]
    active = [i for i, outcome in enumerate(outcomes) if not outcome]
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
        active = []
        for i, action in zip(pending, actions, strict=True):
            game = games[i]
            previous_boxes = game.boxes
            step_game(game, action)
            if deadlock_detection and pushed_into_deadlock(game, previous_boxes):
                outcomes[i] = "死锁"
            else:
                active.append(i)
    return outcomes


def play_level(
    game: Game,
    choose_action: Callable[[bytes, Sequence[int]], int],
    max_steps: int,
    *,
    detect_cycle: bool = True,
    deadlock_detection: bool = True,
) -> str:
    """Play up to max_steps primitive moves. Disable cycle detection for stochastic policies."""
    def choose_one(states, masks):
        return [choose_action(states[0], masks[0])]

    return play_batch(
        [game], choose_one, max_steps, detect_cycle=detect_cycle, deadlock_detection=deadlock_detection,
    )[0]


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="评估 Double DQN 在内置关卡或固定随机地图上的表现")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练保存的检查点")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-steps", type=int, default=256, help="每关角色移动步数上限，含走路和推箱")
    parser.add_argument("--generated", type=int, metavar="N", help="评估 N 个生成关卡，默认内置 30 关")
    parser.add_argument("--max-boxes", type=int, choices=range(1, 5), default=4, help="生成地图的箱数上限")
    parser.add_argument("--seed", type=int, default=2026, help="固定生成地图与随机基线的种子")
    parser.add_argument("--compare-random", action="store_true", help="在相同开局对照均匀随机合法动作")
    parser.add_argument("--deadlock-detection", action=argparse.BooleanOptionalAction, default=True,
                        help="检测到死锁时提前结束评估，默认开启")
    args = parser.parse_args(argv)
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
    deadlock_detection = getattr(args, "deadlock_detection", True)

    def random_action(state: bytes, mask: Sequence[int]) -> int:
        return actions_rng.choice([action for action, legal in enumerate(mask) if legal])

    attempts = Counter()
    wins = Counter()
    random_wins = Counter()
    random_deadlocks = 0
    outcomes = Counter()
    zero_push = 0
    levels = []
    size_counts = {}
    if args.generated is not None and args.max_boxes == 1:
        size_counts = {"small": Counter(), "large": Counter()}
    source = "内置关卡" if args.generated is None else f"生成 {args.generated} 关，最多 {args.max_boxes} 箱"
    print(f"{source}，种子 {args.seed}，设备 {device}，模型 v{version}，每关最多移动 {args.max_steps} 步")
    boards = evaluation_boards(args.generated, args.max_boxes, args.seed)
    for level, (title, rows) in enumerate(boards, start=1):
        game = Game(rows)
        boxes = len(game.boxes)
        outcome = play_level(game, greedy, args.max_steps, deadlock_detection=deadlock_detection)
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
            random_outcome = play_level(baseline, random_action, args.max_steps, detect_cycle=False,
                                        deadlock_detection=deadlock_detection)
            random_wins[boxes] += int(baseline.won)
            random_deadlocks += int(random_outcome == "死锁")
            record["random"] = {"outcome": random_outcome, "moves": baseline.moves, "pushes": baseline.pushes}
        if size_counts:
            width, height = len(rows[0]), len(rows)
            record.update(width=width, height=height)
            group = size_counts["small" if max(width, height) <= SMALL_BOARD_MAX_SIDE else "large"]
            group["attempts"] += 1
            group["wins"] += game.won
            if args.compare_random:
                group["random_wins"] += baseline.won
        levels.append(record)

    def report(label: str, won: int, trials: int, random_won: int) -> dict:
        def score(wins: int) -> str:
            return f"{wins}/{trials}（{wins / trials:.1%}）" if trials else "0/0"

        counts = {"wins": won, "attempts": trials}
        result = f"{label}通关 {score(won)}"
        if args.compare_random:
            result += f" | 随机基线 {score(random_won)}"
            counts["random_wins"] = random_won
        print(result)
        return counts

    by_boxes = {}
    for boxes, count in sorted(attempts.items()):
        by_boxes[boxes] = report(f"{boxes} 箱", wins[boxes], count, random_wins[boxes])
    by_board_size = {}
    for name, counts in size_counts.items():
        bound = f"≤{SMALL_BOARD_MAX_SIDE}" if name == "small" else f"≥{SMALL_BOARD_MAX_SIDE + 1}"
        by_board_size[name] = report(f"长边{bound}格", counts["wins"], counts["attempts"], counts["random_wins"])
    summary = report("总", wins.total(), attempts.total(), random_wins.total())
    summary.update(cycles=outcomes["循环"], deadlocks=outcomes["死锁"], timeouts=outcomes["达到步数上限"],
                   no_legal_actions=outcomes["无可用动作"], zero_push=zero_push)
    print(f"模型循环 {outcomes['循环']} 局，死锁 {outcomes['死锁']} 局，步数耗尽 {outcomes['达到步数上限']} 局，"
          f"无可用动作 {outcomes['无可用动作']} 局，零次推动 {zero_push} 局")
    if args.compare_random:
        summary["random_deadlocks"] = random_deadlocks
        print(f"随机基线死锁 {random_deadlocks} 局")
    path = save_report(args.checkpoint, {
        "checkpoint": args.checkpoint.name,
        "checkpoint_steps": checkpoint_steps, "version": version, "device": device,
        "source": "builtin" if args.generated is None else "generated",
        "seed": args.seed, "generated": args.generated,
        "max_boxes": args.max_boxes if args.generated is not None else None,
        "max_steps": args.max_steps, "compare_random": args.compare_random, "deadlock_detection": deadlock_detection,
        "summary": summary, "by_boxes": by_boxes, "levels": levels,
        **({"by_board_size": by_board_size} if by_board_size else {}),
    })
    print(f"评估报告已保存到 {path}")


if __name__ == "__main__":
    main()
