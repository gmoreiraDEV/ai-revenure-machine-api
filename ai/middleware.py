from __future__ import annotations

import json
from typing import Callable, Optional, Tuple, List, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage

from ai.agent import AgentConfig
from ai.tools import tavily_search


def _dbg(cfg: AgentConfig, *args) -> None:
    if cfg.debug_agent_logs:
        try:
            print(*args, flush=True)
        except Exception:
            pass


def extract_settings_from_messages(messages) -> Tuple[Optional[str], Optional[bool]]:
    """
    Lê a última SystemMessage com JSON {"type":"settings"} e retorna (model_name, use_tavily).
    """
    model_name: Optional[str] = None
    use_tavily: Optional[bool] = None

    for msg in reversed(messages or []):
        try:
            if getattr(msg, "type", None) != "system":
                continue
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue
            data = json.loads(content)
            if isinstance(data, dict) and data.get("type") == "settings":
                if isinstance(data.get("model"), str) and data["model"].strip():
                    model_name = data["model"].strip()
                if isinstance(data.get("use_tavily"), bool):
                    use_tavily = data["use_tavily"]
                break
        except Exception:
            continue

    return model_name, use_tavily


def strip_settings_messages(messages):
    """
    Remove mensagens SystemMessage com JSON {"type":"settings"} antes do LLM.
    """
    cleaned = []
    for msg in messages or []:
        try:
            if getattr(msg, "type", None) == "system":
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and data.get("type") == "settings":
                            continue
                    except Exception:
                        pass
        except Exception:
            pass
        cleaned.append(msg)
    return cleaned


class DynamicSettingsMiddleware(AgentMiddleware):
    """
    Troca dinâmica de modelo e controle de ferramentas por requisição.

    Fonte de verdade (precedência):
      1) runtime.config.configurable.{model_name,use_tavily}
      2) runtime.context.{model_name,use_tavily}
      3) SystemMessage {"type":"settings", ...}
      4) defaults do AgentConfig
    """

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    def _inject_guardrail(self, messages, use_tavily: bool):
        if use_tavily:
            return messages
        try:
            guardrail = SystemMessage(
                content=(
                    "A ferramenta de busca 'tavily_search' está desabilitada para esta requisição. "
                    "Não invente resultados de pesquisa. Se o usuário solicitar pesquisa na web, "
                    "informe que a pesquisa está desabilitada e peça para habilitar Tavily."
                )
            )
            return list(messages) + [guardrail]
        except Exception:
            return messages

    def _resolve_prefs(self, request: ModelRequest) -> Tuple[Optional[str], bool, List[Any]]:
        messages = getattr(request, "messages", None) or []
        model_name, use_tavily = extract_settings_from_messages(messages)

        # runtime overrides
        runtime = getattr(request, "runtime", None)
        try:
            runtime_cfg = getattr(runtime, "config", None)
            if isinstance(runtime_cfg, dict):
                cfg = runtime_cfg.get("configurable", {}) or {}
                if isinstance(cfg.get("model_name"), str) and cfg["model_name"].strip():
                    model_name = cfg["model_name"].strip()
                if isinstance(cfg.get("use_tavily"), bool):
                    use_tavily = cfg["use_tavily"]

            ctx = getattr(runtime, "context", None)
            if isinstance(ctx, dict):
                if isinstance(ctx.get("model_name"), str) and ctx["model_name"].strip():
                    model_name = ctx["model_name"].strip()
                if isinstance(ctx.get("use_tavily"), bool):
                    use_tavily = ctx["use_tavily"]
        except Exception:
            pass

        desired_use_tavily = self.cfg.default_use_tavily if use_tavily is None else use_tavily
        tools = list(getattr(request, "tools", []) or [])
        return model_name, desired_use_tavily, tools

    def _apply_model_tools_messages(self, request: ModelRequest, *, model_name: Optional[str], use_tavily: bool, tools):
        # Tools: remove tavily if disabled
        tavily_name = getattr(tavily_search, "name", "tavily_search")
        tools_filtered = [t for t in tools if getattr(t, "name", None) != tavily_name]
        if use_tavily and any(getattr(t, "name", None) == tavily_name for t in tools):
            tools_filtered.append(tavily_search)

        # Messages: strip settings + guardrail
        cleaned = strip_settings_messages(getattr(request, "messages", []) or [])
        cleaned = self._inject_guardrail(cleaned, use_tavily)

        # Model: if override, create new ChatOpenAI using AgentConfig (no env)
        new_model = getattr(request, "model", None)
        if model_name:
            try:
                new_model = ChatOpenAI(
                    model=model_name,
                    temperature=self.cfg.temperature,
                    openai_api_key=self.cfg.openrouter_api_key,
                    openai_api_base=self.cfg.openrouter_base_url,
                )
            except Exception as e:
                _dbg(self.cfg, f"[SETTINGS] erro ao aplicar modelo '{model_name}': {e}")

        request.model = new_model
        request.messages = cleaned
        request.tools = tools_filtered

        _dbg(
            self.cfg,
            f"[MIDDLEWARE] model={model_name or 'default'} use_tavily={use_tavily} "
            f"tools before={len(tools)} after={len(tools_filtered)}"
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        model_name, use_tavily, tools = self._resolve_prefs(request)
        self._apply_model_tools_messages(request, model_name=model_name, use_tavily=use_tavily, tools=tools)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        model_name, use_tavily, tools = self._resolve_prefs(request)
        self._apply_model_tools_messages(request, model_name=model_name, use_tavily=use_tavily, tools=tools)
        return await handler(request)

    def _desired_use_tavily(self, request) -> bool:
        """
        Para toolcall blocking, re-resolve o use_tavily com base no runtime/state.
        (mantém compat com o fluxo antigo)
        """
        desired = self.cfg.default_use_tavily

        try:
            runtime = getattr(request, "runtime", None)
            runtime_cfg = getattr(runtime, "config", None)
            if isinstance(runtime_cfg, dict):
                cfg = runtime_cfg.get("configurable", {}) or {}
                if isinstance(cfg.get("use_tavily"), bool):
                    desired = cfg["use_tavily"]

            ctx = getattr(runtime, "context", None)
            if isinstance(ctx, dict) and isinstance(ctx.get("use_tavily"), bool):
                desired = ctx["use_tavily"]
        except Exception:
            pass

        try:
            state = getattr(request, "state", None)
            messages = state.get("messages", []) if isinstance(state, dict) else []
            _, sm_tavily = extract_settings_from_messages(messages)
            if isinstance(sm_tavily, bool):
                desired = sm_tavily
        except Exception:
            pass

        return desired

    def _tool_name_from_request(self, request) -> Optional[str]:
        tool = getattr(request, "tool", None)
        if tool is not None:
            name = getattr(tool, "name", None)
            if name:
                return name

        tool_call = getattr(request, "tool_call", None)
        if isinstance(tool_call, dict):
            return tool_call.get("name")
        return getattr(tool_call, "name", None)

    def _tool_call_id_from_request(self, request) -> str:
        tool_call = getattr(request, "tool_call", None)
        if isinstance(tool_call, dict):
            return str(tool_call.get("id") or "toolcall")
        return str(getattr(tool_call, "id", None) or "toolcall")

    def wrap_tool_call(self, request, handler):
        try:
            tavily_name = getattr(tavily_search, "name", "tavily_search")
            if self._tool_name_from_request(request) == tavily_name:
                if not self._desired_use_tavily(request):
                    return ToolMessage(
                        content="Tavily search está desabilitada para esta requisição.",
                        tool_call_id=self._tool_call_id_from_request(request),
                        status="error",
                    )
        except Exception:
            pass
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        try:
            tavily_name = getattr(tavily_search, "name", "tavily_search")
            if self._tool_name_from_request(request) == tavily_name:
                if not self._desired_use_tavily(request):
                    return ToolMessage(
                        content="Tavily search está desabilitada para esta requisição.",
                        tool_call_id=self._tool_call_id_from_request(request),
                        status="error",
                    )
        except Exception:
            pass
        return await handler(request)
