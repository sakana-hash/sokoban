"""Movement, revisit and potential-based rewards."""

from sokoban.engine import DIRECTIONS, Game, Point

from .board import ACTIONS, SIDE

DEFAULT_REVISIT_PENALTY = 0.02


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
