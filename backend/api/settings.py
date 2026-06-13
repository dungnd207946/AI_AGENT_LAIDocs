"""Settings API — read/write config, test LLM connection."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from ..core.config import RerankerConfig, get_settings, reload_settings
from ..services.agent import reset_agent

router = APIRouter(prefix="/api/settings", tags=["settings"])


# ── schemas ─────────────────────────────────────────────────────────

class _MaskedSettings(BaseModel):
    llm: dict[str, Any]
    vlm: dict[str, Any]
    reranker: dict[str, Any]
    port: int


class _SettingsUpdate(BaseModel):
    llm: dict[str, Any] | None = None
    vlm: dict[str, Any] | None = None
    reranker: dict[str, Any] | None = None
    port: int | None = None


class _TestRequest(BaseModel):
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    text: str = "Hello, this is a test."


class _TestRerankerRequest(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    query: str = "What does the document say about installation?"
    documents: list[str] | None = None


# ── helpers ─────────────────────────────────────────────────────────

def _mask(d: dict[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(d, BaseModel):
        d = d.model_dump()
    out = dict(d)
    for key in ("api_key",):
        if out.get(key):
            out[key] = out[key][:4] + "***" if len(out[key]) > 4 else "***"
    return out


# ── routes ──────────────────────────────────────────────────────────

@router.get("/", response_model=_MaskedSettings)
async def read_settings():
    s = get_settings()
    return _MaskedSettings(
        llm=_mask(s.llm),
        vlm=_mask(s.vlm),
        reranker=_mask(s.reranker),
        port=s.port,
    )


@router.put("/", response_model=_MaskedSettings)
async def update_settings(body: _SettingsUpdate):
    s = get_settings()
    if body.llm is not None:
        s.llm = s.llm.model_copy(update=body.llm)
    if body.vlm is not None:
        s.vlm = s.vlm.model_copy(update=body.vlm)
    if body.reranker is not None:
        s.reranker = s.reranker.model_copy(update=body.reranker)
    if body.port is not None:
        s.port = body.port
    s.save_to_file()
    # Reload singleton
    reload_settings()
    # Reset agent if LLM or VLM config changed so it rebuilds with new model(s)
    if body.llm is not None or body.vlm is not None:
        reset_agent()
    s2 = get_settings()
    return _MaskedSettings(
        llm=_mask(s2.llm),
        vlm=_mask(s2.vlm),
        reranker=_mask(s2.reranker),
        port=s2.port,
    )


@router.post("/test-llm")
async def test_llm(body: _TestRequest):
    """Send a tiny chat request to validate LLM credentials (any provider)."""
    from ..core.config import LLMConfig
    from ..services.llm import create_chat_model, is_llm_configured

    s = get_settings()
    cfg = LLMConfig(
        provider=body.provider or s.active_llm.provider,
        base_url=body.base_url or s.active_llm.base_url,
        api_key=body.api_key or s.active_llm.api_key,
        model=body.model or s.active_llm.model,
    )
    if not is_llm_configured(cfg):
        raise HTTPException(
            status_code=400,
            detail="LLM model and credentials (api_key or base_url) are required",
        )
    try:
        model = create_chat_model(cfg, max_tokens=10)
        resp = model.invoke([{"role": "user", "content": "Say hello in one word."}])
        reply = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"success": True, "response": reply}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.post("/test-vlm")
async def test_vlm(body: _TestRequest):
    """Validate VLM credentials with a tiny text chat-completion request."""
    s = get_settings()
    base_url = body.base_url or s.active_vlm.base_url
    api_key = body.api_key or s.active_vlm.api_key
    model = body.model or s.active_vlm.model
    if not all([base_url, api_key, model]):
        raise HTTPException(status_code=400, detail="VLM base_url, api_key, and model are required")
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        reply = resp.choices[0].message.content if resp.choices else ""
        return {"success": True, "response": reply}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.post("/test-reranker")
async def test_reranker(body: _TestRerankerRequest):
    """Validate reranker credentials against a Jina-compatible endpoint."""
    s = get_settings()
    cfg = RerankerConfig(
        enabled=True,
        base_url=body.base_url or s.active_reranker.base_url,
        api_key=body.api_key or s.active_reranker.api_key,
        model=body.model or s.active_reranker.model,
        top_n=min(max(1, len(body.documents or [])), 8) if body.documents else 2,
        candidate_k=s.active_reranker.candidate_k,
        timeout_s=s.active_reranker.timeout_s,
    )
    if not cfg.base_url or not cfg.model:
        raise HTTPException(
            status_code=400,
            detail="Reranker base_url and model are required",
        )

    documents = body.documents or [
        "Installation guide: run the setup command and verify dependencies.",
        "Troubleshooting: if the service does not start, inspect the logs.",
    ]

    payload = {
        "model": cfg.model,
        "query": body.query,
        "documents": documents,
        "top_n": min(max(1, cfg.top_n), len(documents)),
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    try:
        timeout = httpx.Timeout(cfg.timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(cfg.base_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return {"success": False, "error": "Invalid reranker response: missing results"}
        return {
            "success": True,
            "results": [
                {
                    "index": item.get("index"),
                    "score": item.get("relevance_score"),
                }
                for item in results[: payload["top_n"]]
                if isinstance(item, dict)
            ],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
