import argparse
import json
import webbrowser
from threading import Thread

from . import LEVELS, load_level, solve
from .server import make_server


def main() -> None:
    parser = argparse.ArgumentParser(description="推箱子 · 本地游戏与机器求解")
    parser.add_argument(
        "--solve",
        type=int,
        choices=range(1, len(LEVELS) + 1),
        metavar="关卡号",
        help="求解指定关卡并输出 JSON",
    )
    parser.add_argument("--no-browser", action="store_true", help="只启动服务")
    parser.add_argument("--port", type=int, default=8765, help="本机端口（0 为自动选择）")
    args = parser.parse_args()
    if args.solve is not None:
        game = load_level(args.solve)
        actions = solve(game)
        if actions is None:
            parser.exit(1, f"第 {args.solve} 关无解\n")
        for action in actions:
            game.move(action)
        result = {
            "level": args.solve,
            "actions": actions,
            "moves": game.moves,
            "pushes": game.pushes,
            "won": game.won,
        }
        print(json.dumps(result))
        return

    if not 0 <= args.port <= 65535:
        parser.error("端口须在 0 到 65535 之间")
    try:
        server = make_server(args.port)
    except OSError as error:
        parser.exit(1, f"无法开启本地游戏服务（{error}）。可以用 --port 0 自动选择端口。\n")

    with server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"在浏览器中打开 {url} 进入推箱子，按 Ctrl+C 退出服务。", flush=True)
        if not args.no_browser:
            Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
