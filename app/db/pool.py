from __future__ import annotations

from typing import Optional

from psycopg_pool import AsyncConnectionPool

_pool: Optional[AsyncConnectionPool] = None


def init_pool(database_url: str, min_size: int = 1, max_size: int = 10) -> None:
    """
    Cria o pool (open=False). A abertura real acontece em open_pool().

    - database_url vem do Settings (não lemos env aqui).
    - min_size/max_size também vêm do Settings.
    """
    global _pool
    if _pool is not None:
        return

    _pool = AsyncConnectionPool(
        conninfo=database_url,
        open=False,
        min_size=min_size,
        max_size=max_size,
    )


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Did you call init_pool()?")
    return _pool


async def open_pool() -> None:
    pool = get_pool()
    await pool.open()


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
