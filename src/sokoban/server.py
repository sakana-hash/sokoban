import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit

from . import LEVELS, load_level, solve


ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}


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


class GameHandler(BaseHTTPRequestHandler):
    def send_data(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status: int, value: object) -> None:
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_data(status, data, "application/json")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/levels":
            levels = [{"id": i, "title": title} for i, (title, _) in enumerate(LEVELS, 1)]
            self.send_json(HTTPStatus.OK, {"levels": levels})
        elif path in ASSETS:
            name, mime = ASSETS[path]
            content = files("sokoban").joinpath("static", name).read_bytes()
            self.send_data(HTTPStatus.OK, content, mime)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/game":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1024 * 1024:
                raise ValueError("请求正文须为 1 字节至 1 MiB")
            request = json.loads(self.rfile.read(size))
            result = play_request(request)
        except (ValueError, RecursionError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self.send_json(HTTPStatus.OK, result)

    def log_request(self, code="-", size="-") -> None:
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)


def make_server(port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), GameHandler)
