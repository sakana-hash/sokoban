"""Run Double DQN self-play from the repository root."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_OUTPUT = Path("runs/ddqn-v2")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    num_envs: int = 16
    batch_size: int = 64
    replay_size: int = 50_000
    warmup: int = 2_000
    epsilon_steps: int = 150_000
    curriculum_steps: int = 10_000
    max_episode_steps: int = 256
    validation_rooms: int = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Double DQN 在随机关卡中自主试玩")
    parser.add_argument("--steps", type=int, default=250_000, help="累计角色移动步数，包括已完成的步数")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output", type=Path, help=f"新任务默认保存到项目根目录下的 {DEFAULT_OUTPUT}")
    parser.add_argument("--resume", type=Path, help="从完整的 latest.pt 恢复")
    parser.add_argument("--save-every", type=int, default=25_000, help="每多少个环境步保存一次")
    options = {
        "seed": "随机种子",
        "num_envs": "每轮同时采样的环境数",
        "batch_size": "每次梯度更新的样本数",
        "replay_size": "经验池容量",
        "warmup": "开始更新前至少采集的步数",
        "epsilon_steps": "预热后探索率降至 0.05 所需的步数",
        "curriculum_steps": "课程检查间隔（角色移动步数）",
        "max_episode_steps": "单局角色移动步数上限",
        "validation_rooms": "每种箱数使用的固定验证地图数",
    }
    for name, help_text in options.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            help=f"{help_text}，默认 {getattr(Config(), name)}")
    args = parser.parse_args()
    if args.steps < 1 or args.save_every < 1:
        parser.error("--steps 和 --save-every 必须大于 0")

    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        parser.exit(1, "请在服务器的独立训练环境中安装 training/requirements.txt。\n")

    from .runner import run
    from .runtime import checkpoint_version, select_device

    try:
        device = select_device(args.device)
        checkpoint = None
        config = Config()
        output = args.output or (Path(__file__).resolve().parents[1] / DEFAULT_OUTPUT)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            checkpoint_version(checkpoint)
            if "replay" not in checkpoint:
                raise ValueError("续训需要完整的 latest.pt 检查点")
            if args.steps <= checkpoint["steps"]:
                raise ValueError(f"--steps 应大于已完成的 {checkpoint['steps']} 步")
            config = Config(**checkpoint["config"])
            output = args.output or args.resume.parent

        values = asdict(config)
        for name in values:
            override = getattr(args, name)
            if override is not None:
                if checkpoint is not None and override != values[name]:
                    raise ValueError(f"续训时 {name} 必须与检查点一致（{values[name]}）")
                values[name] = override
        config = Config(**values)
        if any(value < 1 for name, value in values.items() if name not in ("seed", "warmup")):
            raise ValueError("环境数、训练批量、经验池和步数参数均须为正整数")
        if config.warmup < 0 or config.batch_size > config.replay_size:
            raise ValueError("预热步数不能为负，经验池须能容纳一个训练批量")
        run(config, device, args.steps, args.save_every, output, checkpoint, args.resume)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
