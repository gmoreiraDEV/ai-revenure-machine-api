from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain.tools import tool
from langchain_tavily import TavilySearch

from app.core.settings import get_settings


def _resolve_tavily_key(runtime: Optional[Dict[str, Any]] = None) -> str:
    """
    Resolve a API key do Tavily com precedência:
      1) runtime.config.configurable.tavily_api_key
      2) runtime.context.tavily_api_key
      3) Settings (TAVILY_API_KEY)
    """
    # 1) runtime.config.configurable
    try:
        if isinstance(runtime, dict):
            cfg = runtime.get("config", {}) or {}
            if isinstance(cfg, dict):
                conf = cfg.get("configurable", {}) or {}
                if isinstance(conf, dict):
                    key = conf.get("tavily_api_key")
                    if isinstance(key, str) and key.strip():
                        return key.strip()
    except Exception:
        pass

    # 2) runtime.context
    try:
        if isinstance(runtime, dict):
            ctx = runtime.get("context", {}) or {}
            if isinstance(ctx, dict):
                key = ctx.get("tavily_api_key")
                if isinstance(key, str) and key.strip():
                    return key.strip()
    except Exception:
        pass

    # 3) Settings fallback
    settings = get_settings()
    key = getattr(settings, "tavily_api_key", None)
    if isinstance(key, str) and key.strip():
        return key.strip()

    raise RuntimeError("TAVILY_API_KEY não configurada (Settings ou runtime).")


@tool
def tavily_search(
    query: str,
    max_results: int = 5,
    include_raw_content: bool = False,
    runtime: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Busca conteúdo na Internet com Tavily.

    Observação: `runtime` pode ser injetado pelo LangChain/LangGraph dependendo do executor.
    Se não vier, cai no Settings.
    """
    api_key = _resolve_tavily_key(runtime)

    client = TavilySearch(
        api_key=api_key,
        max_results=max_results,
        topic="general",
        include_raw_content=include_raw_content,
    )
    result = client.invoke(input=query)

    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)
