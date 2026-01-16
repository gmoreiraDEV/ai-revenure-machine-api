from __future__ import annotations

from typing import Optional, List, Tuple

from psycopg import AsyncConnection

from app.db.pool import get_pool


async def insert_thread(thread_id: str) -> None:
    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                insert into threads (thread_id)
                values (%s)
                on conflict (thread_id) do nothing
                """,
                (thread_id,),
            )
            await conn.commit()


async def get_thread_created_at(thread_id: str) -> Optional[str]:
    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select to_char(created_at at time zone 'utc', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
                  from threads
                 where thread_id = %s
                """,
                (thread_id,),
            )
            row = await cur.fetchone()
            return row[0] if row else None


async def list_threads(limit: int = 50) -> List[Tuple[str, str]]:
    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select thread_id,
                       to_char(created_at at time zone 'utc', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as created
                  from threads
                 order by created_at desc
                 limit %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
            return [(r[0], r[1]) for r in rows]
