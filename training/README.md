# 在 CUDA 服务器上训练 Double DQN

模型从四个方向中选择动作，通过 `Game.move()` 试玩，按实际奖励学习。训练时程序随机生成关卡，内置 30 关留作独立评估。训练和评估全程都不调用求解器。浏览器里的「机器演示」仍使用游戏自带的搜索算法。

## 服务器环境

将包含 `src/` 和 `training/` 的项目复制到服务器的 **50 GB SSD 数据盘**，随后进入项目根目录。可以用 `df -h .` 核对当前目录所在的挂载点。下文的相对路径会把检查点保存在数据盘。

### A100 预装镜像

Ubuntu 22.04 镜像自带的 Python 3.12、PyTorch 2.8.0 和 CUDA 12.8 符合本项目要求。在镜像默认的 Python 环境中直接检查 GPU。[PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/#v280)

```bash
python -c "import sys, torch; print(sys.version, torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
df -h .
```

设备名应包含 A100。首轮采用默认的 16 个采样环境、64 个样本的训练批量和 50,000 条经验的回放池。一个 Python 进程依次推进各环境，并把棋盘批量送入 GPU。训练、日志和评估均可在终端完成，6006 和 6008 端口可保持空闲。使用预装镜像时可以直接进入「验证与启动」。

### 从空白环境安装

对于未预装 PyTorch 的服务器，使用 Python 3.12 或更高版本创建独立环境。

```bash
python3.12 -m venv training/.venv
source training/.venv/bin/activate
nvidia-smi
```

训练的直接依赖只有 PyTorch，版本范围在 `training/requirements.txt`。以下命令安装与该镜像对应的 CUDA 12.8 构建，其他驱动环境可按官方版本表选择。[PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/#v280)

```bash
python -m pip install -r training/requirements.txt 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

## 验证与启动

先运行训练目录的测试。其中端到端测试会用 CPU 做短训练、读取检查点续训、重新加载权重并试玩 30 关。接着用单独的输出目录跑一个短 CUDA 任务。按下面的参数，最后一行日志应显示 `steps=256`、`updates=49`，且 `loss` 是有限值。

```bash
python -m unittest discover -s training/tests -v
python -m training.train --device cuda --steps 256 --warmup 64 --batch-size 32 --output training/runs/smoke
```

以下命令训练 25 万步，随后让模型独立挑战内置 30 关。

```bash
python -m training.train --device cuda --steps 250000 --output training/runs/ddqn
python -m training.evaluate --checkpoint training/runs/ddqn/policy.pt --device cuda
```

每个新任务使用独立的输出目录。日志中的 `steps` 是所有环境合计的动作数，`updates` 是梯度更新次数，`win_rate_last_100` 是最近至多 100 个已结束训练局的通关率，包含探索动作。评估使用固定权重和贪心策略，按关卡及箱子数量报告通关结果，同时区分循环、无可用动作和超时。25 万步是首轮实验预算，效果以实际评估为准。

## 保存与续训

默认每 25,000 个环境步以及训练结束时保存检查点。每个文件先写到同目录的临时文件，再原子替换。`policy.pt` 是评估用的网络权重，`latest.pt` 还保存目标网络、优化器、经验池、各环境的当前局面、随机数状态和训练计数。

```bash
python -m training.train --device cuda --resume training/runs/ddqn/latest.pt --steps 500000
```

`--steps` 始终指累计环境步数。续训从检查点恢复采样和学习参数，可用 `--save-every` 调整保存间隔。

## 训练约定

- **输入与动作**　棋盘补墙到 10×10，四个通道依次表示墙、目标、箱子和玩家。卷积网络保留格子位置，输出上、下、左、右的 Q 值。随机探索、贪心决策和学习目标都只考虑当前能执行的动作。
- **起始局面**　从箱子全部归位的棋盘随机反向拉箱，只保留打乱后的开局。每局的动作均由训练策略产生。默认先生成单箱关，每经过 50,000 步，新开局的箱子数上限增加一个，最多四个，简单关继续参与训练。
- **奖励**　每步扣 0.01 分，已归位箱数的变化量按每箱 1 分计，全部归位另加 10 分。通关或没有可走方向时终止。每局默认最多 256 步，超时时从重置前的实际后继状态估计后续回报。
- **Double DQN**　在线网络选择后继状态下的最佳合法动作，目标网络为这个动作估值。默认 16 个环境在 CPU 上轮流推进，网络批量推理。经验池保存最近 50,000 步，预热 2,000 步后每采集 4 步更新一次，批量为 64。使用 Huber 损失、Adam（学习率 0.0003）和 0.99 的折扣因子，每 500 次梯度更新同步一次目标网络。探索率在预热后用 150,000 步从 1.0 线性降到 0.05。

运行 `python -m training.train --help` 查看可调参数。
