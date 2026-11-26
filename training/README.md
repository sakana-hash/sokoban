# 在 CUDA 服务器上训练 Double DQN

新版 Double DQN 与旧版都使用上、下、左、右四个动作。网络每次选择一个合法方向，角色移动一格，相邻箱子可推动时随之移动一格。绕到箱子后面的每一步同样由网络决定。训练和评估都使用真实游戏规则，不调用求解器。训练地图随机生成，固定生成地图用于验证，内置 30 关用于独立评估。浏览器里的「机器演示」使用游戏自带的搜索算法。

## 服务器环境

将包含 `src/` 和 `training/` 的项目复制到服务器的 **50 GB SSD 数据盘**，随后进入项目根目录。可以用 `df -h .` 核对当前目录所在的挂载点。下文的命令将检查点写入项目根目录的 `runs/`，与训练代码目录 `training/` 分开。

若服务器上已有 `training/runs/`，请在替换 `training/` 前结束正在写入该目录的训练任务，然后从项目根目录运行一次以下命令（Ubuntu）。它会把旧实验逐个迁到 `runs/`，已有的同名实验会留在原处。

```bash
mkdir -p runs && find training/runs -mindepth 1 -maxdepth 1 -exec mv -n {} runs/ \;
find training/runs -mindepth 1 -maxdepth 1 -print
```

若第二条命令列出旧实验，请先给这些实验另取名称并迁到 `runs/`。确认旧目录为空后即可替换 `training/`。

### A100 预装镜像

Ubuntu 22.04 镜像自带的 Python 3.12、PyTorch 2.8.0 和 CUDA 12.8 符合本项目要求。在镜像默认的 Python 环境中直接检查 GPU。[PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/#v280)

```bash
python -c "import sys, torch; print(sys.version, torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
df -h .
```

设备名应包含 A100。首轮采用默认的 16 个采样环境、64 个样本的训练批量和 50,000 条经验的回放池。一个 Python 进程依次推进各环境，并把棋盘批量送入 GPU。训练、日志和评估均可在终端完成，6006 和 6008 端口可保持空闲。使用预装镜像时可以直接评估旧权重或启动新训练。

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

## 用现有权重诊断短循环

以下命令在同一批 200 个固定种子的单箱关卡上评估旧模型。程序自动识别版本 1 的四方向权重。每个开局都另跑一遍均匀随机合法动作，地图生成和随机动作使用独立的随机数流。可以对新权重重复同一条命令以复用地图。

```bash
python -m training.evaluate --device cuda --checkpoint runs/ddqn/policy.pt --generated 200 --max-boxes 1 --seed 2026 --compare-random
```

查看通关率、循环、超时和零次推动局数。每局默认最多让角色走 256 步，含绕到箱子背后的移动。确定性策略回到相同策略输入即记为循环。随机策略可以从重复局面走出，继续运行到通关、无动作或预算用尽。新旧模型和随机基线都只能选择当前合法的上下左右动作。相同种子和地图范围下，两次评估的随机基线应一致。

## 训练新模型

先运行测试，再用独立目录跑短 CUDA 任务。下列短测结束时应有 256 次决策和 49 次梯度更新。

```bash
python -m unittest discover -s training/tests -v
python -m training.train --device cuda --steps 256 --warmup 64 --batch-size 32 --output runs/smoke-v2
```

要检验折返扣分的效果，沿用上一轮 `ddqn-v2-mix` 的随机种子和 25 万步预算，在新目录从头训练。用种子 2027 的同一批 1,000 个单箱开局对照上一轮的通关数、循环数和零推动局数。

```bash
python -m training.train --device cuda --steps 250000 --output runs/ddqn-v2-revisit
python -m training.evaluate --device cuda --checkpoint runs/ddqn-v2-revisit/best.pt --generated 1000 --max-boxes 1 --seed 2027 --compare-random
python -m training.evaluate --device cuda --checkpoint runs/ddqn-v2-revisit/best.pt --compare-random
```

最后一条命令评估内置 30 关。也可以将生成地图命令中的 `--max-boxes` 改为 4，查看 1 到 4 箱的分组结果。25 万次决策是一次实验的预算，效果以相同地图和移动上限下的独立评估为准。

## 模型与课程

- **观测与动作。** 棋盘补墙到 10×10，四个通道分别表示墙、目标、箱子和角色当前位置。空间残差卷积网络输出上下左右四个 Q 值，采样和估值都使用同一组合法动作。每次决策可走一步，或者把紧邻的一个箱子推一格。每局默认最多移动 256 步。
- **距离奖励。** 基础奖励为每步扣 0.01 分、归位箱数的净变化每箱 1 分、通关加 10 分。新版额外加上 `0.99 × Φ(后继局面) − Φ(当前局面)`。第一项是未归位箱子到最近空目标的曼哈顿距离之和除以全部箱数，第二项是角色到最近有利站位的曼哈顿距离。有利站位允许玩家向一个能缩短上述箱子到目标距离的方向推一步。两项距离上限均为 18，权重分别为 −0.10 和 −0.04。终止局面的 Φ 为零。该信号只参与训练奖励，网络仍自行决定每一步。
- **折返扣分。** v2 训练时从开局的角色位置开始记录。普通走动每次踩回已记录的格子，额外扣除 `--revisit-penalty` 指定的分数，默认 0.02。设为 0 可关闭这项扣分。每次推箱后，从玩家的新位置重新记录。续训沿用检查点中的扣分值，旧 v2 检查点缺少该字段时采用 0.02。
- **更新。** 在线网络选定后继局面的最佳合法移动，目标网络给出该移动的估值。默认 16 个环境，经验池容量 50,000，预热 2,000 步后每 4 步更新一次，批量为 64。使用 Huber 损失、Adam（学习率 0.0003）、0.99 的折扣因子，每 500 次更新同步目标网络。
- **单箱开局。** 前 50,000 步，新开局有 75% 的概率采用简化生成方式（用 1 至 2 次合法反向拉箱打乱），其余 25% 采用完整难度生成器。随后完整难度的概率按累计移动步数线性增加，在 125,000 步达到 50%，在 200,000 步达到 75%，此后保持 75%。日志中的 `full_start_probability` 表示此刻开始一局时使用完整生成器的概率。完整生成器也可能产生简单关卡。
- **升难。** 每 10,000 步在完整难度的固定地图上进行贪心验证，1、2、3、4 箱各 32 局，每局最多 256 步。已解锁的每一组均达到 75%（默认至少 24/32）才增加一个箱数等级。升入多箱阶段后全部新开局都使用常规生成器。日志中的 `wins_by_boxes_1_to_4` 是各组验证胜局数，`win_rate_last_100` 是最近结束的训练局含探索的胜率。

## 检查点与续训

检查点在验证后、默认每 25,000 次决策和训练结束时写到任务目录。新版任务至少完成一次验证后才有 `best.pt`。

| 文件 | 用途 |
| --- | --- |
| `best.pt` | 评估四组固定验证地图上总胜局数最高的一版权重，平分保留较早的权重。 |
| `policy.pt` | 评估最近一次保存的在线网络。 |
| `latest.pt` | 续训，恢复模型版本、网络、优化器、经验池、随机数状态、课程和采样中的局面。 |

```bash
python -m training.train --device cuda --resume runs/ddqn-v2-revisit/latest.pt --steps 500000
```

`--steps` 是累计决策目标。恢复任意 v2 检查点后，下一局直接按已保存的累计步数取样。例如，从旧实验的 250,000 步单箱检查点续训时，新开局抽到完整生成器的概率为 75%。已开始的局继续使用原棋盘。版本 1 检查点仍沿用原先按步数升级的课程。

新版 `latest.pt` 会保存每局自上次推箱以来走过的格子。恢复缺少这项记录的旧 v2 检查点时，各局从恢复时角色所在的格子开始记录，经验池中的样本保留采样时的奖励。

## 实验记录

训练在输出到终端的同时，把指标逐行写入检查点旁的 `metrics.jsonl`。每行是一个 JSON 对象，`event` 标明记录类型。

| `event` | 记录内容 |
| --- | --- |
| `start` | 启动时间（UTC）、设备、模型版本、训练配置、起始步数、目标步数和续训来源。 |
| `train` | 累计步数、已结束的训练局数及胜负、更新次数、探索率、损失、最近 100 局胜率和吞吐量。 |
| `validation` | 验证步数、各箱数的通关数、每组关数、当前课程等级和最佳检查点步数。 |

每结束一局训练，通关计入 `episodes_won`，未通关且用尽步数或无合法动作计入 `episodes_lost`。仍在进行的局等到结束时再计数。每条训练记录都满足 `episodes = episodes_won + episodes_lost + episodes_unknown`。`latest.pt` 保存累计数供续训。新实验的 `episodes_unknown` 为 0。从旧检查点恢复时，根据其中最近最多 100 局的结果计入胜负，更早的局归入 `episodes_unknown`。

在原实验目录续训会追加一条 `start` 记录，再继续写指标。指定新的 `--output` 会从恢复点开始建立该目录的日志。旧版检查点只保留存档时的训练状态，此前的完整曲线需从当时保存的终端输出中获取。

以下示例在项目根目录读取每次验证的单箱通关率。

```python
import json

with open("runs/ddqn-v2-revisit/metrics.jsonl", encoding="utf-8") as log:
    for row in map(json.loads, log):
        if row["event"] == "validation":
            rate = row["wins_by_boxes_1_to_4"][0] / row["rooms_per_group"]
            print(row["validation_steps"], f"{rate:.1%}")
```

每次运行 `training.evaluate` 都会在检查点旁的 `evaluations/` 中生成一份独立的 JSON 报告。报告保存评估参数、总体和各箱数的汇总，以及每关的结果、步数和推动次数。使用相同地图参数与种子时，可按关卡编号对照两个模型在哪些关卡改变了结果。已有权重也可以重新评估并保存报告。

## A100 吞吐量

显存约 500 MB 和 GPU 利用率 36% 是单次采样。模型处理 10×10 棋盘，CPU 负责生成地图和执行每步动作，GPU 负责批量推理及梯度更新。用日志中的 `decisions_per_sec` 和 `updates_per_sec` 比较实际吞吐量，同时观察验证通关率。

以下两次短跑在全新的输出目录中进行，均以相同随机种子和训练批量运行，只改变每批环境数。预热后对照几个连续日志区间，比较累计决策和更新次数相同时的用时。

```bash
python -m training.train --device cuda --steps 8192 --num-envs 16 --batch-size 64 --output runs/bench-16x64
python -m training.train --device cuda --steps 8192 --num-envs 64 --batch-size 64 --output runs/bench-64x64
```

一次只改一个参数。若要比较批量 128，可固定环境数再建独立任务。更大批量会同时改变梯度样本数，训练效果仍需用固定地图评估。运行 `python -m training.train --help` 查看可调参数。
