"""LAIDocs sidecar server — FastAPI backend for the Electron desktop app."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# ── UTF-8 handling (critical on Windows) ───────────────────────────
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── ensure project root is on sys.path ─────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import LAIDOCS_HOME, get_settings
from backend.core.database import get_db, init_db
from backend.core.exceptions import LAIDocsError
from backend.core.vault import VAULT_DIR, ensure_assets_dir

# ── CLI arguments ──────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="LAIDocs sidecar server")
parser.add_argument("--dev", action="store_true", help="Run in development mode")
cli_args = parser.parse_args()
DEV_MODE = cli_args.dev

# ── Lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    LAIDOCS_HOME.mkdir(parents=True, exist_ok=True)
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    (LAIDOCS_HOME / "data").mkdir(parents=True, exist_ok=True)
    init_db()

    # Ensure the default "unsorted" (Inbox) folder always exists
    unsorted_dir = VAULT_DIR / "unsorted"
    unsorted_dir.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO folders (path, name) VALUES (?, ?)",
            ("unsorted", "unsorted"),
        )

    # Mount vault assets directory as static files for image serving
    from fastapi.staticfiles import StaticFiles
    assets_path = ensure_assets_dir()
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    from backend.core.telemetry import track_event_sync
    track_event_sync("app_launched")

    print("[sidecar] Server ready")
    yield
    # Shutdown (nothing to clean up yet)


# ── App ────────────────────────────────────────────────────────────

app = FastAPI(title="LAIDocs API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ──────────────────────────────────────

@app.exception_handler(LAIDocsError)
async def laidocs_error_handler(request: Request, exc: LAIDocsError):
    """Convert domain errors to JSON HTTP responses."""
    return JSONResponse(
        status_code=exc.http_status,
        content={"detail": exc.message},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Catch-all handler — log and return 500."""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )


# ── Routers ────────────────────────────────────────────────────────

from backend.api import (
    backup_router,
    settings_router,
    documents_router,
    folders_router,
    chat_router,
    download_router,
)

app.include_router(backup_router)
app.include_router(settings_router)
app.include_router(documents_router)
app.include_router(folders_router)
app.include_router(chat_router)
app.include_router(download_router)


# ── Health check ───────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "dev": DEV_MODE}


# ── Stdin listener (shutdown command) ──────────────────────────────

def _stdin_listener():
    """Background thread that reads stdin for the 'sidecar shutdown' command."""
    try:
        for line in sys.stdin:
            line = line.strip()
            if line == "sidecar shutdown":
                os.kill(os.getpid(), signal.SIGINT)
                break
    except Exception:
        pass  # stdin closed or unavailable


def _start_stdin_listener():
    t = threading.Thread(target=_stdin_listener, daemon=True)
    t.start()


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    config = get_settings()
    _start_stdin_listener()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=config.port,
        log_level="debug" if DEV_MODE else "info",
    )
