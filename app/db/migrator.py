from __future__ import annotations

from pathlib import Path
from typing import List

from psycopg import AsyncConnection

from app.db.pool import get_pool


MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def _ensure_schema_migrations_table(conn: AsyncConnection) -> None:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            create table if not exists schema_migrations (
                version text primary key,
                applied_at timestamptz not null default now()
            );
            """
        )
        await conn.commit()


async def _get_applied_versions(conn: AsyncConnection) -> set[str]:
    async with conn.cursor() as cur:
        await cur.execute("select version from schema_migrations;")
        rows = await cur.fetchall()
        return {r[0] for r in rows}


def _list_migration_files() -> List[Path]:
    if not MIGRATIONS_DIR.exists():
        return []
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    return files


async def run_migrations() -> None:
    """
    Executa migrations SQL versionadas em app/db/migrations/*.sql.

    - Cada arquivo .sql é identificado pelo nome (ex: "0001_create_threads.sql").
    - Só roda migrations ainda não aplicadas (registradas em schema_migrations).
    - Executa em transação por migration.
    """
    pool = get_pool()

    files = _list_migration_files()
    if not files:
        return

    async with pool.connection() as conn:  # type: AsyncConnection
        await _ensure_schema_migrations_table(conn)
        applied = await _get_applied_versions(conn)

        for path in files:
            version = path.name
            if version in applied:
                continue

            sql = path.read_text(encoding="utf-8").strip()
            if not sql:
                # Migration vazia: marca como aplicada e segue
                async with conn.cursor() as cur:
                    await cur.execute(
                        "insert into schema_migrations (version) values (%s);",
                        (version,),
                    )
                    await conn.commit()
                continue

            # Uma transação por migration
            async with conn.cursor() as cur:
                try:
                    await cur.execute("begin;")
                    await cur.execute(sql)
                    await cur.execute(
                        "insert into schema_migrations (version) values (%s);",
                        (version,),
                    )
                    await cur.execute("commit;")
                except Exception:
                    await cur.execute("rollback;")
                    raise
