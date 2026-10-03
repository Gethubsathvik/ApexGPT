"""Optional deployable HTTP inference service."""
from .server import create_app, main

__all__ = ["create_app", "main"]