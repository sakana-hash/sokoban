"""Load a saved policy and run the evaluation command."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .benchmark import run_evaluation


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
    from ..dqn import make_network, mask_batch, state_batch
    from ..runtime import checkpoint_version, select_device

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
