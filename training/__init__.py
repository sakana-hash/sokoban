"""Self-play training tools. Run from the repository root with python -m."""

import sys
from pathlib import Path

# Server checkouts can use the sibling game sources without installing the game.
_game_src = str(Path(__file__).resolve().parent.parent / "src")
if _game_src not in sys.path:
    sys.path.insert(0, _game_src)
