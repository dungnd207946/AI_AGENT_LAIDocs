import { useEffect, useState, useCallback } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { apiGet, apiPost } from "../lib/sidecar";
import { useFolderContext } from "../context/FolderContext";
import { useSidecar } from "../hooks/useSidecar";
import { useUpload, PendingUpload } from "../context/UploadContext";
import FileTree, { FolderNode } from "./FileTree";
import UploadDialog from "./UploadDialog";
import CrawlDialog from "./CrawlDialog";

const getFolderOfDoc = (folders: FolderNode[], docId: string): string | null => {
  for (const f of folders) {
    if (f.documents.some((d) => d.id === docId)) return f.path;
    if (f.children && f.children.length > 0) {
      const found = getFolderOfDoc(f.children, docId);
      if (found) return found;
    }
  }
  return null;
};

// ── SVG Icons ──────────────────────────────────────────────────────
const IconHome = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
    <polyline points="9 22 9 12 15 12 15 22"/>
  </svg>
);

const IconSearch = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/>
    <path d="m21 21-4.3-4.3"/>
  </svg>
);

const IconFolder = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
  </svg>
);

const IconSettings = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3"/>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
  </svg>
);

const IconPlus = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19"/>
    <line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);

const IconCheck = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);

// ── Nav Item ───────────────────────────────────────────────────────
function NavItem({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: "relative",
        display: "flex",
        alignItems: "center",
        gap: 9,
        width: "100%",
        padding: "8px 12px 8px 14px",
        borderRadius: 8,
        fontSize: 13.5,
        fontWeight: active ? 500 : 400,
        color: active ? "var(--text-primary)" : hovered ? "var(--text-secondary)" : "var(--text-muted)",
        background: active ? "var(--accent-subtle)" : hovered ? "var(--surface-hover)" : "transparent",
        border: "none",
        cursor: "pointer",
        transition: "all 0.15s ease",
        textDecoration: "none",
        textAlign: "left",
      }}
    >
      {active && <span className="nav-item-active-bar" />}
      {children}
    </button>
  );
}

const IconRefresh = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 4 23 10 17 10"/>
    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
  </svg>
);

const IconSun = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4"/>
    <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
  </svg>
);

const IconMoon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
);

type Theme = "dark" | "light";

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "light") root.setAttribute("data-theme", "light");
  else root.removeAttribute("data-theme");
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("laidocs-theme");
    return saved === "light" ? "light" : "dark";
  });
  const [hovered, setHovered] = useState(false);

  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem("laidocs-theme", theme);
  }, [theme]);

  const isLight = theme === "light";

  return (
    <button
      onClick={() => setTheme(isLight ? "dark" : "light")}
      title={isLight ? "Switch to dark theme" : "Switch to light theme"}
      aria-label={isLight ? "Switch to dark theme" : "Switch to light theme"}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        flexShrink: 0,
        width: 28,
        height: 28,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 7,
        border: "none",
        background: hovered ? "var(--surface-alt)" : "transparent",
        color: hovered ? "var(--text-muted)" : "var(--text-faint)",
        cursor: "pointer",
        transition: "all 0.15s ease",
      }}
    >
      {isLight ? <IconMoon /> : <IconSun />}
    </button>
  );
}

function ReloadButton() {
  const [spinning, setSpinning] = useState(false);

  const handleReload = () => {
    setSpinning(true);
    setTimeout(() => window.location.reload(), 300);
  };

  return (
    <button
      onClick={handleReload}
      title="Reload app"
      style={{
        flexShrink: 0,
        width: 28,
        height: 28,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 7,
        border: "none",
        background: "transparent",
        color: "var(--text-faint)",
        cursor: "pointer",
        transition: "all 0.15s ease",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = "var(--text-muted)";
        e.currentTarget.style.background = "var(--surface-alt)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = "var(--text-faint)";
        e.currentTarget.style.background = "transparent";
      }}
    >
      <span className={spinning ? "spin" : ""} style={{ display: "flex" }}>
        <IconRefresh />
      </span>
    </button>
  );
}

const STAGE_LABELS: Record<string, { label: string; done: boolean; isError?: boolean }> = {
  uploading:  { label: "uploading…",  done: false },
  uploaded:   { label: "uploaded",    done: true  },
  converting: { label: "converting…", done: false },
  converted:  { label: "converted",   done: true  },
  crawling:   { label: "crawling…",   done: false },
  crawled:    { label: "crawled",     done: true  },
  saving:     { label: "saving…",     done: false },
  saved:      { label: "saved",       done: true  },
  error:      { label: "failed",      done: true, isError: true },
};

function PendingUploadItem({ upload }: { upload: PendingUpload }) {
  const info = STAGE_LABELS[upload.stage] ?? { label: upload.stage, done: false };
  return (
    <div style={{
      padding: "7px 12px 7px 14px",
      borderRadius: 8,
      background: info.isError ? "var(--error-bg)" : info.done ? "var(--success-bg)" : "var(--surface-alt)",
      border: `1px solid ${info.isError ? "rgba(248,113,113,0.15)" : info.done ? "rgba(52,211,153,0.1)" : "var(--border)"}`,
      transition: "all 0.2s ease",
    }}>
      <div style={{
        fontSize: 13,
        color: "var(--text-secondary)",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        marginBottom: 4,
      }}>
        {upload.docTitle || upload.filename}
      </div>
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 5,
        fontSize: 11,
        color: info.isError ? "var(--error)" : info.done ? "var(--success)" : "var(--text-faint)",
        letterSpacing: "0.4px",
      }}>
        {!info.done && !info.isError && (
          <span
            className="spin"
            style={{
              display: "inline-block",
              width: 9,
              height: 9,
              border: "1.5px solid var(--border)",
              borderTopColor: "var(--accent)",
              borderRadius: "50%",
              flexShrink: 0,
            }}
          />
        )}
        {info.isError && (
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        )}
        {info.done && !info.isError && (
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        )}
        {upload.error || info.label}
      </div>
    </div>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export default function Sidebar({ collapsed: _collapsed, onToggleCollapse }: SidebarProps) {
  const { activeFolder, setActiveFolder, refreshFoldersKey, triggerRefreshFolders } =
    useFolderContext();
  const { status } = useSidecar();
  const { pendingUploads } = useUpload();
  const [tree, setTree] = useState<FolderNode[]>([]);
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [newFolderError, setNewFolderError] = useState("");
  const [creating, setCreating] = useState(false);
  const [showNewFile, setShowNewFile] = useState(false);
  const [newFileName, setNewFileName] = useState("");
  const [newFileError, setNewFileError] = useState("");
  const [creatingFile, setCreatingFile] = useState(false);
  const [ctxUploadFolder, setCtxUploadFolder] = useState<string | null>(null);
  const [ctxCrawlFolder, setCtxCrawlFolder] = useState<string | null>(null);
  const location = useLocation();
  const navigate = useNavigate();

  // Extract active doc ID from URL
  const activeDocId = location.pathname.startsWith("/doc/")
    ? location.pathname.replace("/doc/", "")
    : null;

  const fetchTree = useCallback(async () => {
    try {
      const data = await apiGet<FolderNode[]>("/api/folders/tree");
      setTree(data);
    } catch {
      setTree([]);
    }
  }, [refreshFoldersKey]);

  useEffect(() => {
    if (status !== "ready") return;
    fetchTree();
  }, [status, fetchTree]);

  const handleCreateFolder = async () => {
    const name = newFolderName.trim();
    if (!name) { setNewFolderError("Name is required"); return; }
    setNewFolderError("");
    setCreating(true);
    try {
      await apiPost("/api/folders/", { path: name, name });
      setNewFolderName("");
      setShowNewFolder(false);
      triggerRefreshFolders();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to create folder";
      if (msg.includes("409")) {
        setNewFolderError(`A folder named "${name}" already exists`);
      } else {
        setNewFolderError(msg);
      }
    } finally {
      setCreating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleCreateFolder();
    if (e.key === "Escape") {
      setShowNewFolder(false);
      setNewFolderName("");
      setNewFolderError("");
    }
  };

  const handleCreateFile = async () => {
    const name = newFileName.trim();
    if (!name) { setNewFileError("Name is required"); return; }
    setNewFileError("");
    setCreatingFile(true);
    try {
      let targetFolder = activeFolder;
      if (!targetFolder && activeDocId) {
        targetFolder = getFolderOfDoc(tree, activeDocId);
      }
      
      const result = await apiPost<{ id: string }>("/api/documents/create", {
        filename: name,
        folder: targetFolder || "unsorted",
      });
      setNewFileName("");
      setShowNewFile(false);
      triggerRefreshFolders();
      navigate(`/doc/${result.id}`);
    } catch (err) {
      setNewFileError(err instanceof Error ? err.message : "Failed to create file");
    } finally {
      setCreatingFile(false);
    }
  };

  const handleFileKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleCreateFile();
    if (e.key === "Escape") {
      setShowNewFile(false);
      setNewFileName("");
      setNewFileError("");
    }
  };

  const handleFileClick = (docId: string) => {
    setActiveFolder(null);
    navigate(`/doc/${docId}`);
  };

  const isDocsPage = location.pathname === "/";
  const isSearchPage = location.pathname === "/search";
  const isSettingsPage = location.pathname === "/settings";

  return (
    <aside style={{
      width: "100%",
      flexShrink: 0,
      display: "flex",
      flexDirection: "column",
      height: "100%",
      background: "var(--surface)",
      borderRight: "1px solid var(--border)",
    }}>
      {/* Brand */}
      <div style={{
        padding: "16px 14px 14px",
        borderBottom: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <button
          onClick={() => { setActiveFolder(null); navigate("/"); }}
          style={{ background: "none", border: "none", cursor: "pointer", padding: 0, textAlign: "left", display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 0 }}
        >
          <div style={{
            width: 30, height: 30, borderRadius: 9,
            background: "linear-gradient(135deg, var(--accent-subtle), var(--surface-alt))",
            border: "1px solid var(--border-glow)",
            display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0,
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-text)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
            </svg>
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", letterSpacing: "-0.3px", lineHeight: 1 }}>
              Docs Agent
            </div>
            <div style={{ fontSize: 9, color: "var(--text-faint)", marginTop: 3, letterSpacing: "1.4px", textTransform: "uppercase" }}>
              Knowledge Base
            </div>
          </div>
        </button>
        <button
          onClick={onToggleCollapse}
          title="Collapse sidebar"
          className="btn-icon"
          style={{ flexShrink: 0, width: 26, height: 26, marginLeft: 4 }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, overflowY: "auto", padding: "10px 8px 0" }}>
        {/* Home */}
        <NavItem active={isDocsPage && activeFolder === null} onClick={() => { setActiveFolder(null); navigate("/"); }}>
          <IconHome />
          Home
        </NavItem>

        {/* Pending uploads */}
        {pendingUploads.length > 0 && (
          <div style={{ marginTop: 10, marginBottom: 6 }}>
            <div style={{ padding: "0 6px 6px" }}>
              <span className="label-upper" style={{ color: "var(--accent-text)", fontSize: 9 }}>Processing</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {pendingUploads.map((upload) => (
                <PendingUploadItem key={upload.clientId} upload={upload} />
              ))}
            </div>
          </div>
        )}


        {/* Explorer section */}
        <div style={{ marginTop: 20 }}>
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "0 6px 8px",
          }}>
            <span className="label-upper">Explorer</span>
            <div style={{ display: "flex", gap: 2 }}>
              <button
                onClick={() => { setShowNewFile(!showNewFile); setNewFileName(""); setNewFileError(""); setShowNewFolder(false); }}
                className="btn-icon"
                title="New File"
                style={{
                  width: 26, height: 26, borderRadius: 6,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  background: showNewFile ? "var(--accent-subtle)" : "transparent",
                  color: showNewFile ? "var(--accent-text)" : undefined,
                }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="12" y1="11" x2="12" y2="17" />
                  <line x1="9" y1="14" x2="15" y2="14" />
                </svg>
              </button>
              <button
                onClick={() => { setShowNewFolder(!showNewFolder); setNewFolderName(""); setNewFolderError(""); setShowNewFile(false); }}
                className="btn-icon"
                title="New Folder"
                style={{
                  width: 26, height: 26, borderRadius: 6,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  background: showNewFolder ? "var(--accent-subtle)" : "transparent",
                  color: showNewFolder ? "var(--accent-text)" : undefined,
                }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="12" y1="5" x2="12" y2="19"/>
                  <line x1="5" y1="12" x2="19" y2="12"/>
                </svg>
              </button>
            </div>
          </div>

          {/* New file input */}
          {showNewFile && (
            <div style={{ marginBottom: 8, padding: "0 2px", animation: "fadeIn 0.18s ease-out" }}>
              <input
                type="text"
                autoFocus
                value={newFileName}
                onChange={(e) => { setNewFileName(e.target.value); setNewFileError(""); }}
                onKeyDown={handleFileKeyDown}
                placeholder="filename.md"
                className="warp-input"
                style={{ fontSize: 13, padding: "6px 10px", borderRadius: 6 }}
              />
              {newFileError && (
                <p style={{ fontSize: 11, color: "var(--error)", margin: "4px 0 0 2px" }}>{newFileError}</p>
              )}
              <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
                <button
                  onClick={handleCreateFile}
                  disabled={creatingFile}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    fontSize: 12, color: "var(--accent-text)", background: "var(--accent-subtle)",
                    border: "1px solid var(--border-glow)", borderRadius: 6, cursor: "pointer",
                    padding: "4px 10px", transition: "all 0.15s", opacity: creatingFile ? 0.6 : 1,
                  }}
                >
                  {creatingFile ? <span className="spin" style={{ display: "inline-block", width: 10, height: 10, border: "1.5px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%" }} /> : <IconCheck />}
                  Create
                </button>
                <button
                  onClick={() => { setShowNewFile(false); setNewFileName(""); setNewFileError(""); }}
                  style={{ fontSize: 12, color: "var(--text-faint)", background: "none", border: "none", cursor: "pointer", padding: "4px 6px" }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* New folder input */}
          {showNewFolder && (
            <div style={{ marginBottom: 8, padding: "0 2px", animation: "fadeIn 0.18s ease-out" }}>
              <div style={{ position: "relative" }}>
                <input
                  type="text"
                  autoFocus
                  value={newFolderName}
                  onChange={(e) => { setNewFolderName(e.target.value); setNewFolderError(""); }}
                  onKeyDown={handleKeyDown}
                  placeholder="Folder name…"
                  className="warp-input"
                  style={{ fontSize: 13, padding: "6px 10px", borderRadius: 6 }}
                />
              </div>
              {newFolderError && (
                <p style={{ fontSize: 11, color: "var(--error)", margin: "4px 0 0 2px" }}>{newFolderError}</p>
              )}
              <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
                <button
                  onClick={handleCreateFolder}
                  disabled={creating}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    fontSize: 12, color: "var(--accent-text)", background: "var(--accent-subtle)",
                    border: "1px solid var(--border-glow)", borderRadius: 6, cursor: "pointer",
                    padding: "4px 10px", transition: "all 0.15s", opacity: creating ? 0.6 : 1,
                  }}
                >
                  {creating ? <span className="spin" style={{ display: "inline-block", width: 10, height: 10, border: "1.5px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%" }} /> : <IconCheck />}
                  Create
                </button>
                <button
                  onClick={() => { setShowNewFolder(false); setNewFolderName(""); setNewFolderError(""); }}
                  style={{ fontSize: 12, color: "var(--text-faint)", background: "none", border: "none", cursor: "pointer", padding: "4px 6px" }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* File tree */}
          <FileTree
            tree={tree}
            activeDocId={activeDocId}
            onFileClick={handleFileClick}
            activeFolder={activeFolder}
            onFolderClick={(path) => setActiveFolder(path)}
            triggerRefreshFolders={triggerRefreshFolders}
            onUploadToFolder={(folderPath) => setCtxUploadFolder(folderPath)}
            onCrawlToFolder={(folderPath) => setCtxCrawlFolder(folderPath)}
          />
        </div>
      </nav>

      {/* Footer: Settings + Reload */}
      <div style={{
        padding: "8px",
        borderTop: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        gap: 4,
      }}>
        <div style={{ flex: 1 }}>
          <NavItem active={isSettingsPage} onClick={() => navigate("/settings")}>
            <IconSettings />
            Settings
          </NavItem>
        </div>
        <ThemeToggle />
        <ReloadButton />
      </div>

      {/* Context-menu triggered dialogs */}
      <UploadDialog
        open={ctxUploadFolder !== null}
        onClose={() => setCtxUploadFolder(null)}
        onUploadSuccess={triggerRefreshFolders}
        initialFolder={ctxUploadFolder}
      />
      <CrawlDialog
        open={ctxCrawlFolder !== null}
        onClose={() => setCtxCrawlFolder(null)}
        onCrawlSuccess={triggerRefreshFolders}
        initialFolder={ctxCrawlFolder}
      />
    </aside>
  );
}
