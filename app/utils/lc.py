# app/utils/lc.py
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)


def lc_message_to_dict(msg: BaseMessage) -> Dict[str, Any]:
    """
    Converte uma mensagem do LangChain para um formato simples pro frontend.
    """
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, SystemMessage):
        role = "system"
    else:
        role = getattr(msg, "type", "unknown") or "unknown"

    content = getattr(msg, "content", "")
    return {"role": role, "content": content}


def lc_messages_to_list(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Converte lista de mensagens do LangChain para lista de dicts.
    """
    return [lc_message_to_dict(m) for m in (messages or [])]
