"""Generate solvable self-play starts by reversing legal box pushes."""

from random import Random

from sokoban.engine import DIRECTIONS, Game, Point

from .board import SIDE


def _reachable(start: Point, floor: set[Point], boxes: set[Point]) -> set[Point]:
    reached = {start}
    pending = [start]
    while pending:
        x, y = pending.pop()
        for dx, dy in DIRECTIONS.values():
            neighbor = x + dx, y + dy
            if neighbor in floor and neighbor not in boxes and neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


def _pulls(boxes: set[Point], reached: set[Point]) -> list[tuple[Point, Point, Point]]:
    """Enumerate (old box, new box, new player) for a legal reverse pull.

    The player first walks to the new box cell, then steps backward one cell,
    pulling the box along. From the resulting position one forward push undoes it.
    """
    options = []
    for x, y in sorted(boxes):
        for dx, dy in DIRECTIONS.values():
            stand = x + dx, y + dy
            retreat = x + 2 * dx, y + 2 * dy
            if stand in reached and retreat in reached:
                options.append(((x, y), stand, retreat))
    return options


def make_training_game(
    rng: Random, max_boxes: int, *, max_extra_pulls: int | None = None, max_side: int = SIDE,
) -> Game:
    """Return a solvable new room with 1..max_boxes boxes and no action labels."""
    if not 1 <= max_boxes <= 4:
        raise ValueError("max_boxes must be between 1 and 4")
    required_side = max(5, max_boxes + 3)
    if not required_side <= max_side <= SIDE:
        raise ValueError(f"max_side must be between {required_side} and {SIDE}")
    if max_extra_pulls is None:
        max_extra_pulls = 4 * max_boxes
    if max_extra_pulls < 0:
        raise ValueError("max_extra_pulls must be nonnegative")
    count = rng.randint(1, max_boxes)
    width = rng.randint(max(5, count + 3), max_side)
    height = rng.randint(max(5, count + 3), max_side)
    floor = {(x, y) for y in range(1, height - 1) for x in range(1, width - 1)}

    # Reserve three consecutive cells to guarantee at least one legal pull.
    dx, dy = rng.choice(tuple(DIRECTIONS.values()))
    anchor = rng.choice(sorted((x, y) for x, y in floor if (x + 2 * dx, y + 2 * dy) in floor))
    x, y = anchor
    first_stand = x + dx, y + dy
    player = x + 2 * dx, y + 2 * dy
    reserved = {anchor, first_stand, player}

    candidates = sorted(floor - reserved)
    for wall in rng.sample(candidates, rng.randint(0, len(candidates) // 4)):
        floor.remove(wall)
        if len(_reachable(anchor, floor, set())) != len(floor):
            floor.add(wall)

    goals = {anchor, *rng.sample(sorted(floor - reserved), count - 1)}
    boxes = (goals - {anchor}) | {first_stand}
    seen = {frozenset(goals), frozenset(boxes)}
    for _ in range(rng.randint(0, max_extra_pulls)):
        reached = _reachable(player, floor, boxes)
        options = _pulls(boxes, reached)
        rng.shuffle(options)
        for old, new, retreat in options:
            moved = (boxes - {old}) | {new}
            if frozenset(moved) not in seen:
                boxes = moved
                player = retreat
                seen.add(frozenset(boxes))
                break
        else:
            break

    # Walking within this component preserves the forward route to the goals.
    if rng.random() < 0.5:
        player = rng.choice(sorted(_reachable(player, floor, boxes)))
    board = [["#"] * width for _ in range(height)]
    for x, y in floor:
        board[y][x] = "." if (x, y) in goals else " "
    for x, y in boxes:
        board[y][x] = "*" if (x, y) in goals else "$"
    x, y = player
    board[y][x] = "+" if player in goals else "@"
    return Game(tuple("".join(row) for row in board))
