"""Sokoban observations, rewards and procedurally generated self-play starts."""

import sys
from pathlib import Path
from random import Random

# Training runs from a source checkout on the server without installing the game.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sokoban.engine import DIRECTIONS, Game, Point

ACTIONS = ("up", "down", "left", "right")
DEFAULT_REVISIT_PENALTY = 0.02
SIDE = 10
AREA = SIDE * SIDE


def encode(game: Game) -> bytes:
    """Return four consecutive binary planes: walls/padding, goals, boxes, player."""
    planes = bytearray(4 * AREA)
    planes[:AREA] = b"\x01" * AREA
    for x, y in game.floor:
        if x >= SIDE or y >= SIDE:
            raise ValueError(f"board exceeds the {SIDE}x{SIDE} observation")
        planes[y * SIDE + x] = 0
    if any(x >= SIDE or y >= SIDE for x, y in game.walls):
        raise ValueError(f"board exceeds the {SIDE}x{SIDE} observation")
    for x, y in game.goals:
        planes[AREA + y * SIDE + x] = 1
    for x, y in game.boxes:
        planes[2 * AREA + y * SIDE + x] = 1
    x, y = game.player
    planes[3 * AREA + y * SIDE + x] = 1
    return bytes(planes)


def legal_mask(game: Game) -> tuple[bool, ...]:
    legal = game.observation()["legal_actions"]
    return tuple(action in legal for action in ACTIONS)


def step_game(
    game: Game, action: int, visited_since_push: set[Point] | None = None,
    *, revisit_penalty: float = DEFAULT_REVISIT_PENALTY,
) -> float:
    """Advance one legal move, optionally tracking revisits between pushes."""
    if not 0 <= action < len(ACTIONS):
        raise ValueError(f"invalid action index {action}")
    previous = len(game.boxes & game.goals)
    pushes_before = game.pushes
    if not game.move(ACTIONS[action]):
        raise ValueError(f"blocked action {ACTIONS[action]}")
    change = len(game.boxes & game.goals) - previous
    reward = -0.01 + change + (10.0 if game.won else 0.0)
    if visited_since_push is not None:
        if game.pushes != pushes_before:
            visited_since_push.clear()
        elif game.player in visited_since_push:
            reward -= revisit_penalty
        visited_since_push.add(game.player)
    return reward


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def potential(game: Game) -> float:
    """Estimate progress from local geometry, without tracing a path or choosing an action."""
    if game.won:
        return 0.0
    open_goals = game.goals - game.boxes
    max_distance = 2 * (SIDE - 1)
    box_distance = 0
    stand_distance = max_distance
    for box in game.boxes - game.goals:
        nearest = min(_manhattan(box, goal) for goal in open_goals)
        box_distance += nearest
        x, y = box
        for dx, dy in DIRECTIONS.values():
            stand = x - dx, y - dy
            destination = x + dx, y + dy
            if stand not in game.floor or stand in game.boxes:
                continue
            if destination not in game.floor or destination in game.boxes:
                continue
            if min(_manhattan(destination, goal) for goal in open_goals) < nearest:
                stand_distance = min(stand_distance, _manhattan(game.player, stand))
    average_distance = min(max_distance, box_distance / len(game.boxes))
    return -0.10 * average_distance - 0.04 * stand_distance


def shaped_reward(base_reward: float, before: float, after: float, terminal: bool, gamma: float) -> float:
    """Use zero potential at true terminals and the actual successor at a time limit."""
    return base_reward + gamma * (0.0 if terminal else after) - before


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


def make_training_game(rng: Random, max_boxes: int, *, max_extra_pulls: int | None = None) -> Game:
    """Return a solvable new room with 1..max_boxes boxes and no action labels."""
    if not 1 <= max_boxes <= 4:
        raise ValueError("max_boxes must be between 1 and 4")
    if max_extra_pulls is None:
        max_extra_pulls = 4 * max_boxes
    if max_extra_pulls < 0:
        raise ValueError("max_extra_pulls must be nonnegative")
    count = rng.randint(1, max_boxes)
    width = rng.randint(max(5, count + 3), SIDE)
    height = rng.randint(max(5, count + 3), SIDE)
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
