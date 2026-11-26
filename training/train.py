"""Run Double DQN self-play from the repository root."""

import argparse
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from .env import DEFAULT_REVISIT_PENALTY


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
    revisit_penalty: float = DEFAULT_REVISIT_PENALTY
    deadlock_penalty: float = -1.0
    small_board_curriculum: bool = True
    model_version: int = 2
    deadlock_detection: bool = True
    anneal_easy_starts: bool = True


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Double DQN 在随机关卡中自主试玩")
    parser.add_argument("--steps", type=int, default=250_000, help="累计角色移动步数，包括已完成的步数")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--channels-last", action="store_true",
                        help="训练时使用 channels_last 卷积内存布局，可在续训时切换，默认关闭")
    parser.add_argument("--model-version", type=int, choices=(2, 3),
                        help="新训练默认使用 v2，v3 的参数量为 v2 的两倍。续训自动读取检查点版本")
    parser.add_argument("--output", type=Path, help="新任务默认保存到项目根目录下的 runs/ddqn-v<版本号>")
    parser.add_argument("--resume", type=Path, help="从完整的 latest.pt 恢复")
    parser.add_argument("--save-every", type=int, default=25_000, help="每多少个环境步保存一次")
    parser.add_argument("--profile", action="store_true", help="采集一个训练性能窗口，保存各阶段用时和时间线")
    parser.add_argument("--profile-start", type=int, metavar="STEP",
                        help="从包含该累计环境步的采样批次开始分析，默认在预热后采集")
    parser.add_argument("--profile-rounds", type=int, metavar="N",
                        help="最多分析多少轮环境推进和梯度更新，默认 16，遇到日志或存档等边界提前结束")
    parser.add_argument("--eval-after-train", action="store_true",
                        help="训练完成后用默认参数评估 best.pt，尚无最佳权重时使用 policy.pt")
    parser.add_argument("--revisit-penalty", type=float,
                        help=f"训练中每次折返的额外扣分，默认 {DEFAULT_REVISIT_PENALTY}，0 表示关闭")
    parser.add_argument("--deadlock-penalty", type=float,
                        help=f"推入死锁时加到奖励上的值，默认 {Config.deadlock_penalty}，设为 0 仍结束该局")
    parser.add_argument("--small-board-curriculum", action=argparse.BooleanOptionalAction, default=None,
                        help="单箱训练前期优先抽取 5–7 格棋盘，新训练默认开启")
    parser.add_argument("--deadlock-detection", action=argparse.BooleanOptionalAction, default=None,
                        help="确认死锁时扣分并终止该局，默认开启")
    parser.add_argument("--anneal-easy-starts", action=argparse.BooleanOptionalAction, default=None,
                        help="逐步降低单箱简化开局的概率，关闭时固定为 75%%")
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
    if not args.profile and (args.profile_start is not None or args.profile_rounds is not None):
        parser.error("--profile-start 和 --profile-rounds 需要配合 --profile")
    if args.profile_start is not None and not 0 <= args.profile_start < args.steps:
        parser.error("--profile-start 必须在 0（含）和 --steps（不含）之间")
    if args.profile_rounds is not None and args.profile_rounds < 1:
        parser.error("--profile-rounds 必须大于 0")
    if args.revisit_penalty is not None and (not isfinite(args.revisit_penalty) or args.revisit_penalty < 0):
        parser.error("--revisit-penalty 必须为非负有限数")
    if args.deadlock_penalty is not None and (not isfinite(args.deadlock_penalty) or args.deadlock_penalty > 0):
        parser.error("--deadlock-penalty 必须为非正有限数")

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
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            version = checkpoint_version(checkpoint)
            if "replay" not in checkpoint:
                raise ValueError("续训需要完整的 latest.pt 检查点")
            if args.steps <= checkpoint["steps"]:
                raise ValueError(f"--steps 应大于已完成的 {checkpoint['steps']} 步")
            saved_config = dict(checkpoint["config"], model_version=version)
            # Earlier runs sampled regular board sizes throughout training.
            saved_config.setdefault("small_board_curriculum", False)
            config = Config(**saved_config)

        values = asdict(config)
        for name in values:
            override = getattr(args, name)
            if override is not None:
                if checkpoint is not None and override != values[name]:
                    raise ValueError(f"续训时 {name} 必须与检查点一致（{values[name]}）")
                values[name] = override
        config = Config(**values)
        run_directory = Path(__file__).resolve().parents[1] / "runs" / f"ddqn-v{config.model_version}"
        output = args.output or (args.resume.parent if args.resume else run_directory)
        if any(values[name] < 1 for name in options if name not in ("seed", "warmup")):
            raise ValueError("环境数、训练批量、经验池和步数参数均须为正整数")
        if config.warmup < 0 or config.batch_size > config.replay_size:
            raise ValueError("预热步数不能为负，经验池须能容纳一个训练批量")
        profile_start = args.profile_start
        if args.profile and profile_start is None:
            warm = max(config.warmup, config.batch_size)
            profile_start = warm if warm < args.steps else 0
        policy_path = run(config, device, args.steps, args.save_every, output, checkpoint, args.resume,
                          profile_start=profile_start, profile_rounds=args.profile_rounds or 16,
                          channels_last=args.channels_last)
        if args.eval_after_train:
            from .evaluate import main as evaluate

            print(f"训练完成，使用 {policy_path} 运行默认评估", flush=True)
            evaluation_args = ["--checkpoint", str(policy_path)]
            if not config.deadlock_detection:
                evaluation_args.append("--no-deadlock-detection")
            evaluate(evaluation_args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
