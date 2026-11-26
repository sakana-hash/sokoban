# 推箱子

30 个可自由选择的关卡，从单箱入门逐步增加到四箱调度。人可以在浏览器里操作，程序可以通过 Python 接口读取棋盘、移动角色并判断胜负。内置求解器可从当前局面接手。整个游戏仅使用 Python 标准库和本地静态文件，无需联网、训练模型或安装 PyTorch。

## 开始游戏

使用 Python 3.12 或更高版本，在项目根目录运行以下命令。也可以在 PyCharm 中直接运行 `main.py`。

```sh
python3.12 main.py
```

浏览器会打开本机游戏页面。用方向键、WASD 或屏幕上的方向按钮移动，把所有箱子推到目标点即可过关。每次只能推动一个箱子，按 Z 撤销一步，按 R 重开本关。点击「机器演示」可观看求解过程，按 Esc 停止后可以继续手动玩。使用同一浏览器和地址再次打开游戏时，会恢复上次的进度。

只启动服务可运行 `python3.12 main.py --no-browser`，然后打开终端中显示的地址。默认端口是 8765，端口被占用时可加 `--port 0` 自动选择。按 Ctrl+C 退出服务。

## 让机器玩

在命令行求解任意一关（编号 1–30），会输出包含方向序列、移动步数、推动次数和通关状态的 JSON。

```sh
python3.12 main.py --solve 30
```

编写自己的策略时，可在项目根目录用 `PYTHONPATH=src python3.12` 启动解释器。下面的代码加载第 8 关，把求解器返回的动作逐步交给游戏引擎。

```python
from sokoban import load_level, solve

game = load_level(8)
print("\n".join(game.observation()["board"]))
actions = solve(game)
if actions is not None:
    for action in actions:
        game.move(action)
print(game.won, game.moves, game.pushes)
```

`game.move(action)` 接受 `up`、`down`、`left`、`right`，返回本次移动是否成功。撞墙或推不动箱子都不会增加步数。`game.undo()` 撤销一步，`game.reset()` 重开本关。`solve(game)` 保留输入局面，无解时返回 `None`，已通关时返回空列表。

`game.observation()` 返回可直接转成 JSON 的状态快照。其中 `legal_actions` 列出当前能走的方向，`player`、`boxes`、`goals` 提供角色、木箱和目标的坐标。左上角为 `(0, 0)`，横坐标向右增加，纵坐标向下增加。

棋盘使用标准推箱子字符。墙为 `#`，箱子为 `$`，目标为 `.`，角色为 `@`。箱子和角色站在目标上时分别显示为 `*` 和 `+`。

## 验证

```sh
python3.12 -m unittest discover -s tests
```

测试逐关求解并回放全部动作，也检查受阻移动、撤销和从中途局面接手求解。

## 训练神经网络

独立的 Double DQN 训练程序使用随机生成的关卡自主试玩，支持 CUDA 训练、断点续训和内置 30 关评估。服务器安装和运行步骤见 [训练说明](training/README.md)。
