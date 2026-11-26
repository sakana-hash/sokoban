"""Four-way actions, board dimensions and observation encoding."""

from sokoban.engine import Game

ACTIONS = ("up", "down", "left", "right")
SIDE = 10
AREA = SIDE * SIDE
CHANNELS = 4


def encode(game: Game) -> bytes:
    """Return four consecutive binary planes: walls/padding, goals, boxes, player."""
    planes = bytearray(CHANNELS * AREA)
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
