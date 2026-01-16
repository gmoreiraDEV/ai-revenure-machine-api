from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from app.db.threads import (
    get_thread_created_at,
    insert_thread,
    list_threads,
)
from app.models.schemas import (
    RunRequest,
    RunResponse,
    RunResult,
    ThreadObj,
    ThreadSearchRequest,
)
from app.utils.lc import lc_messages_to_list


router = APIRouter(tags=["threads"])


def convert_to_lc_messages(raw: List[Dict[str, Any]]) -> List[BaseMessage]:
    """Traduz objetos vindos do frontend para mensagens do LangChain."""
    msgs: List[BaseMessage] = []
    for message in raw:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            content = str(content)

        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        elif role == "system":
            msgs.append(SystemMessage(content=content))
    return msgs


def build_run_config(thread_id: str, body: RunRequest) -> Dict[str, Any]:
    """Monta config enviando thread_id e overrides opcionais."""
    configurable: Dict[str, Any] = {"thread_id": thread_id}
    if body.config and isinstance(body.config.configurable, dict):
        configurable.update(body.config.configurable)
    return {"configurable": configurable}


def chunk_to_text(chunk: Any) -> str:
    """Extrai string utilizável a partir de pedaços do modelo."""
    if isinstance(chunk, AIMessageChunk):
        content = chunk.content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if isinstance(content, str):
            return content

    if hasattr(chunk, "content"):
        value = getattr(chunk, "content")
        if isinstance(value, str):
            return value

    return str(chunk)


def sse_payload(data: Dict[str, Any]) -> str:
    """Formato básico text/event-stream."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def get_graph_or_500(request: Request):
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    return graph


def get_checkpointer_or_500(request: Request):
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is None:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")
    return checkpointer


@router.post("/threads")
async def create_thread(request: Request) -> ThreadObj:
    """Cria um identificador novo para conversas no banco."""
    thread_id = str(uuid.uuid4())
    await insert_thread(thread_id)
    created = await get_thread_created_at(thread_id)
    return ThreadObj(thread_id=thread_id, created_at=created, values={"messages": []})


@router.post("/threads/search")
async def search_threads(request: Request, req: ThreadSearchRequest) -> List[ThreadObj]:
    """Lista threads recentes para permitir seleção no frontend."""
    rows = await list_threads(limit=req.limit or 50)
    return [ThreadObj(thread_id=t, created_at=ts, values={"messages": []}) for t, ts in rows]


@router.get("/threads/{thread_id}")
async def get_thread(request: Request, thread_id: str) -> ThreadObj:
    """Retorna histórico salvo para um thread específico."""
    checkpointer = get_checkpointer_or_500(request)

    created = await get_thread_created_at(thread_id)
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        tup = await checkpointer.aget_tuple(cfg)
        msgs: List[BaseMessage] = []
        if tup and tup.checkpoint:
            msgs = tup.checkpoint.get("channel_values", {}).get("messages", []) or []
    except Exception:
        msgs = []

    return ThreadObj(
        thread_id=thread_id,
        created_at=created,
        values={"messages": lc_messages_to_list(msgs)},
    )


@router.post("/threads/{thread_id}/runs/wait", response_model=RunResponse)
async def run_and_wait(request: Request, thread_id: str, body: RunRequest) -> RunResponse:
    """Fluxo síncrono: aguarda o LangGraph concluir e só então responde."""
    graph = get_graph_or_500(request)
    checkpointer = get_checkpointer_or_500(request)

    in_msgs = convert_to_lc_messages([m.model_dump() for m in body.input.messages])
    cfg = build_run_config(thread_id, body)

    await graph.ainvoke({"messages": in_msgs}, config=cfg)

    tup = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    msgs: List[BaseMessage] = []
    if tup and tup.checkpoint:
        msgs = tup.checkpoint.get("channel_values", {}).get("messages", []) or []

    return RunResponse(result=RunResult(messages=lc_messages_to_list(msgs)))


@router.post("/threads/{thread_id}/runs/stream")
async def run_and_stream(request: Request, thread_id: str, body: RunRequest):
    """Fluxo assíncrono: envia SSE com tokens parciais e resumo final."""
    graph = get_graph_or_500(request)
    checkpointer = get_checkpointer_or_500(request)

    in_msgs = convert_to_lc_messages([m.model_dump() for m in body.input.messages])
    cfg = build_run_config(thread_id, body)

    async def event_iterator():
        try:
            async for event in graph.astream_events({"messages": in_msgs}, config=cfg):
                if event.get("event") == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    text = chunk_to_text(chunk) if chunk is not None else ""
                    if not text:
                        continue
                    yield sse_payload({"event": "chunk", "thread_id": thread_id, "text": text})

            tup = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
            msgs: List[BaseMessage] = []
            if tup and tup.checkpoint:
                msgs = tup.checkpoint.get("channel_values", {}).get("messages", []) or []

            yield sse_payload(
                {"event": "final", "thread_id": thread_id, "messages": lc_messages_to_list(msgs)}
            )
            yield sse_payload({"event": "done", "thread_id": thread_id})
        except Exception as exc:
            yield sse_payload({"event": "error", "detail": str(exc), "thread_id": thread_id})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_iterator(), media_type="text/event-stream", headers=headers)
