"""Sokoban rules shared by the human interface and machine players."""

from __future__ import annotations

from collections.abc import Sequence


Point = tuple[int, int]
DIRECTIONS: dict[str, Point] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def advance(point: Point, delta: Point) -> Point:
    return point[0] + delta[0], point[1] + delta[1]


class Game:
    """A single level. Coordinates use (x, y), starting at the top left."""

    def __init__(self, rows: Sequence[str]) -> None:
        if isinstance(rows, str):
            raise ValueError("Pass a sequence of board rows, e.g. text.splitlines()")
        players: list[Point] = []
        boxes: set[Point] = set()
        goals: set[Point] = set()
        walls: set[Point] = set()
        floor: set[Point] = set()
        self._terrain: list[str] = []
        for y, row in enumerate(rows):
            terrain_row = []
            for x, tile in enumerate(row):
                point = (x, y)
                if tile not in "# .$@*+":
                    raise ValueError(f"Invalid tile {tile!r} at {point}")
                if tile == "#":
                    walls.add(point)
                else:
                    floor.add(point)
                if tile in "@+":
                    players.append(point)
                if tile in "$*":
                    boxes.add(point)
                if tile in ".*+":
                    goals.add(point)
                    terrain_row.append(".")
                else:
                    terrain_row.append("#" if tile == "#" else " ")
            self._terrain.append("".join(terrain_row))
        if len(players) != 1:
            raise ValueError("A level must contain exactly one player")
        if not boxes or len(boxes) != len(goals):
            raise ValueError("A level needs an equal, nonzero number of boxes and goals")

        self.floor = frozenset(floor)
        self.walls = frozenset(walls)
        self.goals = frozenset(goals)
        self._start_player = players[0]
        self._start_boxes = frozenset(boxes)
        self._history: list[tuple[Point, frozenset[Point], int]] = []
        self.reset()

    def reset(self) -> None:
        self.player = self._start_player
        self.boxes = self._start_boxes
        self.pushes = 0
        self._history.clear()

    @property
    def moves(self) -> int:
        return len(self._history)

    @property
    def won(self) -> bool:
        return self.boxes == self.goals

    def _can_move(self, delta: Point) -> bool:
        target = advance(self.player, delta)
        if self.won or target not in self.floor:
            return False
        if target not in self.boxes:
            return True
        beyond = advance(target, delta)
        return beyond in self.floor and beyond not in self.boxes

    def move(self, action: str) -> bool:
        """Apply one direction. A wall, blocked box or finished level returns False."""
        if action not in DIRECTIONS:
            raise ValueError(f"Unknown direction {action!r}")
        delta = DIRECTIONS[action]
        if not self._can_move(delta):
            return False
        self._history.append((self.player, self.boxes, self.pushes))
        self.player = advance(self.player, delta)
        if self.player in self.boxes:
            beyond = advance(self.player, delta)
            self.boxes = (self.boxes - {self.player}) | {beyond}
            self.pushes += 1
        return True

    def undo(self) -> bool:
        if not self._history:
            return False
        self.player, self.boxes, self.pushes = self._history.pop()
        return True

    def observation(self) -> dict:
        """Return a detached JSON-serializable snapshot using standard Sokoban tiles."""
        board = [list(row) for row in self._terrain]
        for x, y in self.boxes:
            board[y][x] = "*" if (x, y) in self.goals else "$"
        x, y = self.player
        board[y][x] = "+" if self.player in self.goals else "@"
        return {
            "board": ["".join(row) for row in board],
            "player": list(self.player),
            "boxes": [list(point) for point in sorted(self.boxes)],
            "goals": [list(point) for point in sorted(self.goals)],
            "moves": self.moves,
            "pushes": self.pushes,
            "won": self.won,
            "legal_actions": [name for name, delta in DIRECTIONS.items() if self._can_move(delta)],
        }
