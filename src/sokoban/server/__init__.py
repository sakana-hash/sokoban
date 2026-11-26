"""Public interface for the local browser game server."""

from .api import play_request
from .http import ASSETS, GameHandler, make_server

__all__ = ["ASSETS", "GameHandler", "make_server", "play_request"]
