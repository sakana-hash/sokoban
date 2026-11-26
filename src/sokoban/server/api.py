"""Apply game API requests to an independent game instance."""

from .. import load_level, solve


def play_request(request: object) -> dict:
    """Rebuild the position from a level and its move history for one client."""
    if not isinstance(request, dict):
        raise ValueError("请求须为 JSON 对象")
    game = load_level(request.get("level"))
    actions = request.get("actions", [])
    command = request.get("command", "state")
    if not isinstance(actions, list):
        raise ValueError("actions 须为方向数组")
    if command not in ("state", "solve"):
        raise ValueError("未知指令")
    for action in actions:
        if action not in ("up", "down", "left", "right"):
            raise ValueError("方向须为 up、down、left 或 right")
        game.move(action)
    result = game.observation()
    if command == "solve":
        result["solution"] = solve(game)
    return result
