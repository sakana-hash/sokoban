"""Evaluate a greedy Double DQN policy on the 30 built-in levels."""

import argparse
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from .env import encode, legal_mask, step_game
from sokoban import Game, LEVELS, load_level


def play_level(
    game: Game,
    choose_action: Callable[[bytes, tuple[bool, ...]], int],
    max_steps: int,
) -> str:
    """Run a deterministic, memoryless policy until the outcome is known."""
    seen = set()
    for _ in range(max_steps):
        if game.won:
            return "通关"
        position = (game.player, game.boxes)
        if position in seen:
            return "循环"
        seen.add(position)
        mask = legal_mask(game)
        if not any(mask):
            return "无可用动作"
        step_game(game, choose_action(encode(game), mask))
    return "通关" if game.won else "达到步数上限"


def main() -> None:
    parser = argparse.ArgumentParser(description="使用冻结的 Double DQN 模型试玩全部 30 关")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练保存的检查点")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-steps", type=int, default=256, help="每关最多移动的步数")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps 必须大于 0")
    if not args.checkpoint.is_file():
        parser.error(f"找不到检查点 {args.checkpoint}")

    try:
        import torch
    except ModuleNotFoundError:
        parser.exit(1, "请在已安装训练依赖的环境中运行评估。\n")
    from .dqn import QNetwork, state_batch
    from .runtime import select_device

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1 or "online" not in checkpoint:
        parser.error("检查点需要包含版本号 1 和在线网络权重 online")
    model = QNetwork()
    model.load_state_dict(checkpoint["online"])
    del checkpoint
    model.to(device).eval()

    def greedy(state: bytes, mask: tuple[bool, ...]) -> int:
        values = model(state_batch([state], device))[0]
        allowed = torch.tensor(mask, dtype=torch.bool, device=device)
        return int(values.masked_fill(~allowed, -torch.inf).argmax().item())

    attempts = Counter()
    wins = Counter()
    print(f"设备 {device}，每关步数上限 {args.max_steps}")
    with torch.inference_mode():
        for level, (title, _) in enumerate(LEVELS, start=1):
            game = load_level(level)
            boxes = len(game.boxes)
            outcome = play_level(game, greedy, args.max_steps)
            print(f"{level:02d} {title} | {boxes} 箱 | {outcome} | {game.moves} 步 {game.pushes} 推")
            attempts[boxes] += 1
            wins[boxes] += int(game.won)
    for boxes, count in sorted(attempts.items()):
        print(f"{boxes} 箱通关 {wins[boxes]}/{count}（{wins[boxes] / count:.1%}）")
    print(f"总通关 {wins.total()}/{attempts.total()}（{wins.total() / attempts.total():.1%}）")


if __name__ == "__main__":
    main()
