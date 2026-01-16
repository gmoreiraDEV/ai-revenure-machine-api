from __future__ import annotations

from contextlib import AsyncExitStack

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.settings import get_settings
from ai.agent import AgentConfig, build_graph


def build_agent_graph(checkpointer: BaseCheckpointSaver):
    settings = get_settings()
    cfg = AgentConfig(
        debug_agent_logs=settings.debug_agent_logs,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
        default_model_name=settings.effective_model_name,
        default_use_tavily=settings.effective_use_tavily,
    )
    return build_graph(cfg=cfg, checkpointer=checkpointer)


async def open_checkpointer(database_url: str) -> tuple[AsyncExitStack, AsyncPostgresSaver]:
    """Cria e mantém um AsyncPostgresSaver ativo até o fechamento."""
    stack = AsyncExitStack()
    cm = AsyncPostgresSaver.from_conn_string(database_url)
    saver = await stack.enter_async_context(cm)
    return stack, saver
