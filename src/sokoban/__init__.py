"""推箱子的公共 Python 接口。关卡编号从 1 开始。"""

from .engine import Game
from .levels import LEVELS
from .solver import solve

__all__ = ["Game", "LEVELS", "load_level", "solve"]


def load_level(level: int = 1) -> Game:
    """Create an independent game at the start of the selected level."""
    if type(level) is not int or not 1 <= level <= len(LEVELS):
        raise ValueError(f"关卡编号须为 1 到 {len(LEVELS)} 之间的整数")
    return Game(LEVELS[level - 1][1])


def main() -> None:
    from .cli import main as run

    run()
