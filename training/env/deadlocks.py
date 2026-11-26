"""Terminal wall-corner and frozen-square deadlocks."""

from sokoban.engine import Game, Point


def is_deadlocked(game: Game, *, moved_box: Point | None = None) -> bool:
    """Check for a fixed-wall corner or a frozen 2x2 containing an off-goal box.

    After a push from a live position only the newly occupied square can create
    either pattern. Floor membership also handles the outer edges of the board.
    """
    off_goal = game.boxes - game.goals
    candidates = game.boxes if moved_box is None else (moved_box,)
    for x, y in candidates:
        if (x, y) in off_goal:
            horizontal_wall = (x - 1, y) not in game.floor or (x + 1, y) not in game.floor
            vertical_wall = (x, y - 1) not in game.floor or (x, y + 1) not in game.floor
            if horizontal_wall and vertical_wall:
                return True
        for dx in (-1, 1):
            for dy in (-1, 1):
                square = ((x, y), (x + dx, y), (x, y + dy), (x + dx, y + dy))
                if not off_goal.isdisjoint(square) and all(
                    cell not in game.floor or cell in game.boxes for cell in square
                ):
                    return True
    return False


def pushed_into_deadlock(game: Game, previous_boxes: frozenset[Point]) -> bool:
    """Check the destination of the single box displaced by a legal move."""
    if game.boxes == previous_boxes:
        return False
    destination, = game.boxes - previous_boxes
    return is_deadlocked(game, moved_box=destination)
