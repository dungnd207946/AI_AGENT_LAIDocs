"""LAIDocs API -- all routers."""

from .backup import router as backup_router
from .chat import router as chat_router
from .downloads import router as download_router
from .folders import router as folders_router
from .settings import router as settings_router
from .documents import documents_router

__all__ = [
    "backup_router",
    "settings_router",
    "documents_router",
    "folders_router",
    "chat_router",
    "download_router",
]
