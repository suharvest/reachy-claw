"""SQLite-backed persistence for daily interaction data and diaries."""

from .db import (
    Database,
    open_default,
    DEFAULT_DB_PATH,
)

__all__ = ["Database", "open_default", "DEFAULT_DB_PATH"]