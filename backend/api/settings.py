"""Settings API — read/write config, test LLM connection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.config import get_settings, reload_settings
from ..services.agent import reset_agent

router = APIRouter(prefix="/api/settings", tags=["settings"])


# ── schemas ─────────────────────────────────────────────────────────

class _MaskedSettings(BaseModel):
    llm: dict[str, Any]
    port: int


class _SettingsUpdate(BaseModel):
    llm: dict[str, Any] | None = None
    port: int | None = None


class _TestRequest(BaseModel):
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    text: str = "Hello, this is a test."


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
        port=s.port,
    )


@router.put("/", response_model=_MaskedSettings)
async def update_settings(body: _SettingsUpdate):
    s = get_settings()
    if body.llm is not None:
        s.llm = s.llm.model_copy(update=body.llm)
    if body.port is not None:
        s.port = body.port
    s.save_to_file()
    # Reload singleton
    reload_settings()
    # Reset agent if LLM config changed so it rebuilds with new model
    if body.llm is not None:
        reset_agent()
    s2 = get_settings()
    return _MaskedSettings(
        llm=_mask(s2.llm),
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
