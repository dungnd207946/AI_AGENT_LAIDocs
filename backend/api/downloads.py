from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core.config import LAIDOCS_HOME

router = APIRouter(tags=["download"])

DOWNLOADS_DIR = LAIDOCS_HOME / "downloads"


def _resolve_download_path(filename: str) -> Path:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md downloads are supported.")
    return DOWNLOADS_DIR / safe_name


@router.get("/download/{filename}")
async def download_export(filename: str):
    """Serve a generated markdown export file from the downloads folder."""
    download_path = _resolve_download_path(filename)
    if not download_path.exists() or not download_path.is_file():
        raise HTTPException(status_code=404, detail="Download file not found.")
    return FileResponse(
        path=str(download_path),
        media_type="text/markdown",
        filename=download_path.name,
    )
