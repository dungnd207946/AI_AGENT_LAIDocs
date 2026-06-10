# VLM Image-Reading Tool — Design

**Date:** 2026-06-10
**Status:** Approved (pending spec review)

## Goal

Add a tool that lets the chat agent "read" images embedded in a document's
Markdown. When the agent encounters an image reference (`![Image N](/assets/...)`)
in retrieved context and the user's question concerns it, the agent calls a new
`read_image` tool. The tool uses a **separate VLM** (e.g. `qwen3-vl-plus`,
OpenAI-compatible vision API) to answer a prompt the agent itself composes.

The VLM is configured exactly like the existing LLM: a new **VLM** tab in
Settings with `base_url`, `api_key`, `model`, plus `.env` defaults. The system
calls the VLM through the same `init_chat_model` path as the LLM.

## Non-Goals

- No automatic image discovery — the agent must pass the image path it extracts
  from the retrieved context.
- No image-based test in the VLM Settings tab (text test only).
- No changes to document conversion / image extraction (already produces
  `![Image N](/assets/<doc_id>_N.png)` refs and saves PNGs to the vault).

## Decisions (from brainstorming)

1. **Image location:** Tool takes `image_path` + `prompt`. The agent extracts the
   `/assets/...` ref from retrieved Markdown context and passes it explicitly.
2. **VLM config:** Mirror the LLM fully — a `vlm: LLMConfig` field plus
   `default_vlm_*` env defaults and an `active_vlm` fallback property.
3. **VLM connection test:** Text-only prompt (same shape as `test-llm`).

## Architecture

```
Agent (DeepAgents)
├── retrieve_context tool   (existing — returns section text incl. image refs)
└── read_image tool         (NEW — VLM-powered)
        │
        ├── resolve /assets/<file> → vault assets dir / <file>
        ├── read file → base64 data URI
        ├── _create_model(settings.active_vlm)   ← shared factory
        └── invoke VLM with [image_url(data URI), prompt] → text answer
```

## Components

### 1. Config — `backend/core/config.py`

- Add to `Settings`:
  - `vlm: LLMConfig = LLMConfig()`
  - `default_vlm_base_url: str = ""`
  - `default_vlm_api_key: str = ""`
  - `default_vlm_model: str = ""`
- Add property `active_vlm` mirroring `active_llm`:
  ```python
  @property
  def active_vlm(self) -> LLMConfig:
      return LLMConfig(
          base_url=self.vlm.base_url or self.default_vlm_base_url,
          api_key=self.vlm.api_key or self.default_vlm_api_key,
          model=self.vlm.model or self.default_vlm_model,
      )
  ```
- `load_from_file` already filters to known keys, so old configs stay
  backward-compatible (missing `vlm` → default empty config).

### 2. Model factory — `backend/services/agent.py`

Refactor `_create_model` to accept an `LLMConfig` directly so it can build either
model:

```python
def _create_model(cfg: LLMConfig):
    return init_chat_model(
        model=cfg.model,
        model_provider="openai",
        base_url=cfg.base_url or None,
        api_key=cfg.api_key or "sk-placeholder",
        max_retries=3,
        timeout=120,
    )
```

Update the existing call site: `model = _create_model(settings.active_llm)`.
Add import of `LLMConfig` from `..core.config`.

### 3. `read_image` tool — `backend/services/agent.py`

```python
@tool
def read_image(image_path: str, prompt: str) -> str:
    """Read an image embedded in the document and answer a question about it.

    Use this when retrieved context contains an image reference like
    ![Image N](/assets/...) and the user's question is about that image
    (a chart, diagram, figure, scanned table, etc.).

    Args:
        image_path: The image reference from the document context,
            e.g. "/assets/<doc_id>_1.png".
        prompt: A precise question/instruction for reading the image.
    """
```

Behaviour:
- Read `settings` from `_tool_context_var` (already populated per request). If
  missing → `"Error: Document context not configured."`
- Validate `settings.active_vlm` has `base_url` + `model`; else return a clear
  message: `"Error: VLM is not configured. Set it in Settings → VLM."`
- Resolve path: strip a leading `/assets/`, take the basename (defensive against
  path traversal), join with the vault assets directory. Reuse the vault assets
  path helper used by the converter (`core/vault`); confirm the exact accessor
  during implementation. If the file doesn't exist → return an error message.
- Read bytes, base64-encode, build a `data:image/png;base64,...` URI.
- `model = _create_model(settings.active_vlm)`; invoke with a single human
  message containing both an `image_url` part (the data URI) and a `text` part
  (the prompt). Use the LangChain multimodal content-block format:
  ```python
  from langchain_core.messages import HumanMessage
  resp = model.invoke([HumanMessage(content=[
      {"type": "text", "text": prompt},
      {"type": "image_url", "image_url": {"url": data_uri}},
  ])])
  return resp.content
  ```
- Wrap the invoke in try/except → return `f"Error reading image: {exc}"` so the
  agent can recover gracefully.

Register the tool: `tools=[retrieve_context, read_image]` in `create_deep_agent`.

### 4. SOUL prompt — `backend/services/agent.py`

Add a short rule under Core Rules / a new "Reading Images" note:

> When retrieved context contains an image reference `![Image N](/assets/...)`
> and the question concerns that image (chart, diagram, figure, scanned table),
> call `read_image` with that exact path and a precise prompt. Only read images
> that appear in the retrieved document context — never invent paths. Treat the
> VLM's answer as part of the document content (still document-grounded).

### 5. Settings API — `backend/api/settings.py`

- `_MaskedSettings`: add `vlm: dict[str, Any]`.
- `_SettingsUpdate`: add `vlm: dict[str, Any] | None = None`.
- `read_settings`: include `vlm=_mask(s.vlm)`.
- `update_settings`: if `body.vlm is not None`, `s.vlm = s.vlm.model_copy(update=body.vlm)`;
  call `reset_agent()` when **either** `llm` or `vlm` changed (agent rebuilds with
  fresh model + tool config).
- New route `POST /api/settings/test-vlm`: same logic as `test-llm` but reads
  `s.active_vlm` defaults and sends a one-word text prompt. (Text-only test.)

### 6. Frontend — `src/pages/Settings.tsx`

- `ServiceConfig` unchanged (reused). `SettingsData` add `vlm: ServiceConfig`.
- Add `"vlm"` to the `Tab` type and a tab entry with an icon (e.g. an "eye"/image
  SVG — `IconVLM`).
- Add `vlmTest` state + `testVlm` callback hitting `/api/settings/test-vlm`.
- Render a `ServiceSection` for `activeTab === "vlm"` (title e.g.
  "Vision Model (OpenAI-compatible API)", placeholder model `qwen3-vl-plus`),
  bound to `settings.vlm`.
- Extend `save` payload + `isDirty` check to include `vlm` fields (same pattern
  as `llm`, but **no** default base_url injection unless desired — keep symmetric
  with llm; decide a sensible placeholder during implementation).

## Data Flow (happy path)

1. User asks a question about a figure in the doc.
2. Agent calls `retrieve_context` → gets section text including
   `![Image 2](/assets/<doc_id>_2.png)`.
3. Agent decides the answer needs the image, calls
   `read_image("/assets/<doc_id>_2.png", "What does this bar chart show?")`.
4. Tool resolves to disk file, base64-encodes, invokes `active_vlm`.
5. VLM returns a description; agent incorporates it (citing the section/image)
   into a document-grounded answer.

## Error Handling

| Condition | Result |
|-----------|--------|
| No tool context | `"Error: Document context not configured."` |
| VLM not configured | `"Error: VLM is not configured. Set it in Settings → VLM."` |
| Image file missing | `"Error: Image <path> not found."` |
| VLM call fails | `"Error reading image: <exc>"` |

All are returned as tool strings so the agent degrades gracefully rather than
crashing the stream.

## Testing

- **Config:** `active_vlm` falls back to `default_vlm_*` when `vlm` empty; round-trips
  through `save_to_file` / `load_from_file`; old config without `vlm` loads fine.
- **Tool:** path resolution (`/assets/x.png` → vault file), missing-file message,
  unconfigured-VLM message. VLM invoke can be mocked to assert the multimodal
  message shape.
- **API:** `test-vlm` returns `{success, response|error}`; `update_settings` with
  `vlm` triggers `reset_agent`.
- **Frontend:** manual — VLM tab renders, Test connection works, Save persists
  `vlm`, dirty state toggles.

## Open Items (resolve during implementation)

- Exact vault assets-dir accessor in `core/vault` (confirm function name).
- Whether to inject a default base_url placeholder for VLM in the frontend (kept
  symmetric with LLM; lean toward no auto-default to avoid surprises).
