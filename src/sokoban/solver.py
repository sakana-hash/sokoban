"""Search over box pushes and expand the resulting plan into individual moves."""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count
from math import inf

from .engine import DIRECTIONS, Game, Point, advance


State = tuple[frozenset[Point], Point]


def _behind(point: Point, delta: Point) -> Point:
    return point[0] - delta[0], point[1] - delta[1]


def _goal_distances(floor: frozenset[Point], goal: Point) -> dict[Point, int]:
    """Minimum box pushes to this goal, ignoring other boxes and player travel."""
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        box = queue.popleft()
        for delta in DIRECTIONS.values():
            previous = _behind(box, delta)
            stand = _behind(previous, delta)
            if previous in floor and stand in floor and previous not in distances:
                distances[previous] = distances[box] + 1
                queue.append(previous)
    return distances


def _walks(floor: frozenset[Point], boxes: frozenset[Point], player: Point) -> dict:
    """Map the reachable area and keep one shortest walk to every square."""
    parents = {player: None}
    queue = deque([player])
    while queue:
        square = queue.popleft()
        for action, delta in DIRECTIONS.items():
            neighbor = advance(square, delta)
            if neighbor in floor and neighbor not in boxes and neighbor not in parents:
                parents[neighbor] = (square, action)
                queue.append(neighbor)
    return parents


def _actions(floor: frozenset[Point], state: State, parents: dict) -> list[str]:
    reverse_actions = []
    while state in parents:
        previous, push = parents[state]
        reverse_actions.append(push)
        stand = _behind(state[1], DIRECTIONS[push])
        boxes, player = previous
        walks = _walks(floor, boxes, player)
        while stand != player:
            stand, step = walks[stand]
            reverse_actions.append(step)
        state = previous
    reverse_actions.reverse()
    return reverse_actions


def solve(game: Game) -> list[str] | None:
    """Return a legal plan from the current position, or None if it is unsolvable."""
    if game.won:
        return []
    distances = [_goal_distances(game.floor, goal) for goal in sorted(game.goals)]

    @lru_cache(maxsize=16384)
    def estimate(boxes: frozenset[Point]) -> float:
        # Match each box to a different goal. For large custom levels use a
        # cheaper lower bound to avoid exponential work inside each estimate.
        costs = [[dist.get(box, inf) for dist in distances] for box in boxes]
        if len(costs) > 8:
            return sum(min(row) for row in costs)
        totals = {0: 0}
        for row in costs:
            updated = {}
            for mask, total in totals.items():
                for index, cost in enumerate(row):
                    bit = 1 << index
                    if mask & bit or cost == inf:
                        continue
                    chosen = mask | bit
                    updated[chosen] = min(updated.get(chosen, inf), total + cost)
            totals = updated
        return totals.get((1 << len(costs)) - 1, inf)

    start = (game.boxes, game.player)
    initial = estimate(game.boxes)
    if initial == inf:
        return None
    order = count()
    pending = [(initial, initial, next(order), 0, start)]
    best = {start: 0}
    parents = {}
    expanded = set()
    while pending:
        _, _, _, pushes, state = heappop(pending)
        if pushes != best[state]:
            continue
        boxes, player = state
        if boxes == game.goals:
            return _actions(game.floor, state, parents)

        reachable = _walks(game.floor, boxes, player)
        # Any player square in the same free region permits identical pushes.
        region = (boxes, min(reachable))
        if region in expanded:
            continue
        expanded.add(region)
        for box in sorted(boxes):
            for action, delta in DIRECTIONS.items():
                destination = advance(box, delta)
                if destination not in game.floor or destination in boxes:
                    continue
                if _behind(box, delta) not in reachable:
                    continue
                new_boxes = (boxes - {box}) | {destination}
                successor = (new_boxes, box)
                cost = pushes + 1
                if cost >= best.get(successor, inf):
                    continue
                remaining = estimate(new_boxes)
                if remaining == inf:
                    continue
                best[successor] = cost
                parents[successor] = (state, action)
                heappush(pending, (cost + remaining, remaining, next(order), cost, successor))
    return None
