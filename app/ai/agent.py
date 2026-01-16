from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from ai.middleware import DynamicSettingsMiddleware
from ai.tools import tavily_search


def _sp_today_str() -> str:
    return datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y")


@dataclass(frozen=True)
class AgentConfig:
    # Logs
    debug_agent_logs: bool = False

    # Provider (OpenRouter via compat OpenAI)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Defaults (com override do Studio aplicado antes de criar o config)
    default_model_name: str = "google/gemini-2.5-flash"
    default_use_tavily: bool = False
    default_system_prompt: str = ""
    temperature: float = 0.2

    # Summarization
    summary_model_name: str = "google/gemini-2.0-flash-001"
    max_tokens_before_summary: int = 10000
    messages_to_keep: int = 12

    @staticmethod
    def default_prompt() -> str:
        return (
            f"Você é um copiloto que auxilia o time. Hoje é {_sp_today_str()} (fuso de São Paulo). "
            "Quando perguntarem a data de hoje, responda usando essa data de forma direta. "
            "Se pedirem a hora exata, responda que só dispõe da data."
        )


def _dbg(cfg: AgentConfig, *args) -> None:
    if cfg.debug_agent_logs:
        try:
            print(*args, flush=True)
        except Exception:
            pass


def _ensure_tools(
    tools: Optional[Iterable[BaseTool]],
) -> List[BaseTool]:
    """
    Mantém o comportamento seguro: sempre registrar tavily_search
    para evitar 'unknown tool' mesmo quando não for usado.
    """
    final_tools: List[BaseTool] = list(tools or [])
    try:
        has = any(getattr(t, "name", None) == getattr(tavily_search, "name", "tavily_search") for t in final_tools)
        if not has:
            final_tools.append(tavily_search)
    except Exception:
        final_tools.append(tavily_search)
    return final_tools


def create_agent_graph(
    *,
    cfg: AgentConfig,
    model_name: str,
    system_prompt: str,
    use_tavily: bool,
    tools: Optional[Iterable[BaseTool]] = None,
    temperature: Optional[float] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """
    Factory: monta e retorna o agente (Runnable) pronto para LangGraph.
    """
    if not cfg.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY não definido (AgentConfig.openrouter_api_key).")

    agent_tools = _ensure_tools(tools)

    llm = ChatOpenAI(
        model=model_name,
        temperature=cfg.temperature if temperature is None else temperature,
        openai_api_key=cfg.openrouter_api_key,
        openai_api_base=cfg.openrouter_base_url,
    )

    _dbg(cfg, f"[AGENT] model={model_name} use_tavily={use_tavily} tools={len(agent_tools)}")

    middlewares = [
        DynamicSettingsMiddleware(cfg),
        SummarizationMiddleware(
            model=ChatOpenAI(
                model=cfg.summary_model_name,
                temperature=0.0,
                openai_api_key=cfg.openrouter_api_key,
                openai_api_base=cfg.openrouter_base_url,
            ),
            max_tokens_before_summary=cfg.max_tokens_before_summary,
            messages_to_keep=cfg.messages_to_keep,
        ),
    ]

    return create_agent(
        model=llm,
        tools=agent_tools,
        system_prompt=system_prompt,
        state_schema=AgentState,
        middleware=middlewares,
        checkpointer=checkpointer,
    )


def build_graph(
    *,
    cfg: AgentConfig,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_tavily: Optional[bool] = None,
    tools: Optional[Iterable[BaseTool]] = None,
    temperature: Optional[float] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """
    Helper: cria o grafo com defaults do config, permitindo overrides.
    """
    final_model = model_name or cfg.default_model_name
    final_prompt = system_prompt or (cfg.default_system_prompt or AgentConfig.default_prompt())
    final_use_tavily = cfg.default_use_tavily if use_tavily is None else use_tavily

    return create_agent_graph(
        cfg=cfg,
        model_name=final_model,
        system_prompt=final_prompt,
        use_tavily=final_use_tavily,
        tools=tools,
        temperature=temperature,
        checkpointer=checkpointer,
    )
