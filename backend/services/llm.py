"""Provider-agnostic LLM factory.

Single source of truth for turning an ``LLMConfig`` into a LangChain chat
model. Supports OpenAI-compatible endpoints (incl. local Ollama / LM Studio),
Google Gemini, and Anthropic — selected via ``LLMConfig.provider``.

Centralising this here means retrieval, summarisation, and the agent all share
identical provider handling, and adding a provider is a one-line change to
``_PROVIDER_MAP`` instead of touching every call site.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ..core.config import LLMConfig

# Map our short provider names → LangChain ``model_provider`` identifiers.
_PROVIDER_MAP = {
    "": "openai",  # empty defaults to OpenAI-compatible
    "openai": "openai",
    "gemini": "google_genai",
    "google": "google_genai",
    "google_genai": "google_genai",
    "anthropic": "anthropic",
    "claude": "anthropic",
}


def normalize_provider(provider: str | None) -> str:
    """Resolve a user-facing provider name to a LangChain provider id."""
    return _PROVIDER_MAP.get((provider or "").strip().lower(), "openai")


def is_llm_configured(cfg: LLMConfig) -> bool:
    """Return True when ``cfg`` has enough to make a call.

    OpenAI-compatible endpoints are considered configured when a ``base_url``
    is present (covers local servers that need no key). Hosted providers
    (Gemini, Anthropic) require an ``api_key``. All require a model.
    """
    if not cfg.model:
        return False
    provider = normalize_provider(cfg.provider)
    if provider == "openai":
        # base_url OR api_key is enough (local server vs. official OpenAI)
        return bool(cfg.base_url or cfg.api_key)
    return bool(cfg.api_key)


def create_chat_model(
    cfg: LLMConfig,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = 120,
    max_retries: int = 3,
    **kwargs,
) -> BaseChatModel:
    """Build a LangChain chat model for the given config and provider.

    Extra keyword arguments are forwarded to the underlying model constructor.
    """
    provider = normalize_provider(cfg.provider)

    params: dict = {
        "model": cfg.model,
        "model_provider": provider,
        "max_retries": max_retries,
    }
    if timeout is not None:
        params["timeout"] = timeout
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    if provider == "openai":
        # base_url is optional (official OpenAI vs. local/compatible server).
        params["base_url"] = cfg.base_url or None
        params["api_key"] = cfg.api_key or "sk-placeholder"
    else:
        # Gemini / Anthropic: pass api_key (LangChain maps it to the provider).
        # base_url is ignored — these providers use their own endpoints.
        if cfg.api_key:
            params["api_key"] = cfg.api_key

    params.update(kwargs)
    return init_chat_model(**params)
