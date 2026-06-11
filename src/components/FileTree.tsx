import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { apiPut, apiDelete } from "../lib/sidecar";

// ── Types ─────────────────────────────────────────────────────────

const UNSORTED_FOLDER = "unsorted";
const UNSORTED_DISPLAY_NAME = "General";

export interface DocNode {
  id: string;
  title: string;
  filename: string;
  source_type: string;
}

export interface FolderNode {
  path: string;
  name: string;
  parent_path: string | null;
  document_count: number;
  children: FolderNode[];
  documents: DocNode[];
}

interface FileTreeProps {
  tree: FolderNode[];
  activeDocId: string | null;
  onFileClick: (docId: string) => void;
  activeFolder: string | null;
  onFolderClick: (path: string) => void;
  triggerRefreshFolders?: () => void;
  onUploadToFolder?: (folderPath: string) => void;
  onCrawlToFolder?: (folderPath: string) => void;
}

interface ContextMenuState {
  x: number;
  y: number;
  target: { type: 'file' | 'folder', id: string, name: string, path: string };
}

// ── Persistence ───────────────────────────────────────────────────

const STORAGE_KEY = "laidocs-tree-expanded";

function loadExpanded(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return new Set(JSON.parse(raw));
  } catch { /* ignore */ }
  return new Set();
}

function saveExpanded(set: Set<string>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...set]));
  } catch { /* ignore */ }
}

// ── SVG Icons ─────────────────────────────────────────────────────

const IconChevron = ({ expanded }: { expanded: boolean }) => (
  <svg
    width="10"
    height="10"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{
      transition: "transform 0.15s ease",
      transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
      flexShrink: 0,
    }}
  >
    <polyline points="9 18 15 12 9 6" />
  </svg>
);

const IconFolderOpen = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    <path d="M2 10h20" />
  </svg>
);

const IconFolderClosed = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
  </svg>
);

const IconFile = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>
);

const IconGlobe = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="2" y1="12" x2="22" y2="12" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
  </svg>
);

const IconGeneral = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="12 2 2 7 12 12 22 7 12 2" />
    <polyline points="2 12 12 17 22 12" />
    <polyline points="2 17 12 22 22 17" />
  </svg>
);

// ── Indent Guide ──────────────────────────────────────────────────

const INDENT_PX = 20;

function IndentGuides({ depth }: { depth: number }) {
  if (depth === 0) return null;
  return (
    <>
      {Array.from({ length: depth }, (_, i) => (
        <span
          key={i}
          style={{
            position: "absolute",
            left: 8 + i * INDENT_PX,
            top: 0,
            bottom: 0,
            width: 1,
            background: "var(--border)",
          }}
        />
      ))}
    </>
  );
}

// ── Rename Input ──────────────────────────────────────────────────

function RenameInput({
  initialValue,
  depth,
  onSubmit,
  onCancel
}: {
  initialValue: string;
  depth: number;
  onSubmit: (v: string) => void;
  onCancel: () => void;
}) {
  const [val, setVal] = useState(initialValue);
  const submitted = useRef(false);

  const handleSubmit = (v: string) => {
    if (submitted.current) return;
    submitted.current = true;
    onSubmit(v);
  };

  const handleCancel = () => {
    if (submitted.current) return;
    submitted.current = true;
    onCancel();
  };

  return (
    <div style={{ paddingLeft: 8 + depth * INDENT_PX, paddingRight: 8, paddingBottom: 2, paddingTop: 2, display: "flex" }}>
      <input
        autoFocus
        value={val}
        onChange={(e) => setVal(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            handleSubmit(val);
          }
          if (e.key === "Escape") {
            e.preventDefault();
            handleCancel();
          }
        }}
        onBlur={() => handleSubmit(val)}
        className="warp-input"
        style={{ width: "100%", fontSize: 13, padding: "2px 6px", borderRadius: 4 }}
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}

// ── File label ────────────────────────────────────────────────────
// The display title is stored without an extension, so re-attach the
// extension from the filename (e.g. "test" + ".md" → "test.md").
function fileLabel(doc: DocNode): string {
  const name = doc.title || doc.filename;
  const dot = doc.filename.lastIndexOf(".");
  const ext = dot > 0 ? doc.filename.slice(dot) : "";
  if (ext && !name.toLowerCase().endsWith(ext.toLowerCase())) {
    return name + ext;
  }
  return name;
}

// ── TreeFile ──────────────────────────────────────────────────────

function TreeFile({
  doc,
  depth,
  isActive,
  onClick,
  onContextMenu,
  isRenaming,
  onRenameSubmit,
  onRenameCancel
}: {
  doc: DocNode;
  depth: number;
  isActive: boolean;
  onClick: () => void;
  onContextMenu: (e: React.MouseEvent, target: ContextMenuState['target']) => void;
  isRenaming: boolean;
  onRenameSubmit: (id: string, v: string, type: 'file' | 'folder') => void;
  onRenameCancel: () => void;
}) {
  const [hovered, setHovered] = useState(false);

  if (isRenaming) {
    return (
      <RenameInput
        initialValue={doc.title || doc.filename}
        depth={depth}
        onSubmit={(v) => onRenameSubmit(doc.id, v, 'file')}
        onCancel={onRenameCancel}
      />
    );
  }

  return (
    <button
      onClick={onClick}
      onContextMenu={(e) => {
        e.preventDefault();
        onContextMenu(e, { type: 'file', id: doc.id, name: doc.title || doc.filename, path: doc.id });
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: "relative",
        display: "flex",
        alignItems: "center",
        gap: 6,
        width: "100%",
        padding: "3px 8px 3px 0",
        paddingLeft: 8 + depth * INDENT_PX,
        border: "none",
        borderRadius: 4,
        fontSize: 13,
        fontWeight: isActive ? 500 : 400,
        fontFamily: "inherit",
        color: isActive ? "var(--text-primary)" : hovered ? "var(--text-secondary)" : "var(--text-muted)",
        background: isActive ? "var(--accent-subtle)" : hovered ? "var(--surface-hover)" : "transparent",
        cursor: "pointer",
        textAlign: "left",
        transition: "color 0.12s, background 0.12s",
        lineHeight: "22px",
        minHeight: 26,
      }}
    >
      <IndentGuides depth={depth} />
      {isActive && <span className="nav-item-active-bar" />}
      <span style={{ color: doc.source_type === "url" ? "var(--accent-text)" : "var(--text-faint)", flexShrink: 0, display: "flex" }}>
        {doc.source_type === "url" ? <IconGlobe /> : <IconFile />}
      </span>
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {fileLabel(doc)}
      </span>
    </button>
  );
}

// ── TreeFolder ────────────────────────────────────────────────────

function TreeFolder({
  folder,
  depth,
  expanded,
  onToggle,
  activeDocId,
  onFileClick,
  expandedSet,
  onToggleFolder,
  activeFolder,
  onFolderClick,
  onContextMenu,
  renameTarget,
  onRenameSubmit,
  onRenameCancel
}: {
  folder: FolderNode;
  depth: number;
  expanded: boolean;
  onToggle: () => void;
  activeDocId: string | null;
  onFileClick: (docId: string) => void;
  expandedSet: Set<string>;
  onToggleFolder: (path: string) => void;
  activeFolder: string | null;
  onFolderClick: (path: string) => void;
  onContextMenu: (e: React.MouseEvent, target: ContextMenuState['target']) => void;
  renameTarget: { id: string, type: 'file' | 'folder' } | null;
  onRenameSubmit: (id: string, v: string, type: 'file' | 'folder') => void;
  onRenameCancel: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const isUnsorted = folder.path === UNSORTED_FOLDER;
  const isRenaming = !isUnsorted && renameTarget?.type === 'folder' && renameTarget.id === folder.path;
  const displayName = isUnsorted ? UNSORTED_DISPLAY_NAME : folder.name;

  return (
    <div>
      {isRenaming ? (
        <RenameInput
          initialValue={folder.name}
          depth={depth}
          onSubmit={(v) => onRenameSubmit(folder.path, v, 'folder')}
          onCancel={onRenameCancel}
        />
      ) : (
        <button
          onClick={(e) => {
            onToggle();
            onFolderClick(folder.path);
          }}
          onContextMenu={(e) => {
            e.preventDefault();
            onContextMenu(e, { type: 'folder', id: folder.path, name: folder.name, path: folder.path });
          }}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          style={{
            position: "relative",
            display: "flex",
            alignItems: "center",
            gap: 5,
            width: "100%",
            padding: "3px 8px 3px 0",
            paddingLeft: 8 + depth * INDENT_PX,
            border: "none",
            borderRadius: 4,
            fontSize: 13,
            fontWeight: 500,
            fontFamily: "inherit",
            color: (folder.path === activeFolder) ? "var(--text-primary)" : hovered ? "var(--text-primary)" : "var(--text-secondary)",
            background: (folder.path === activeFolder) ? "var(--surface-alt)" : hovered ? "var(--surface-hover)" : "transparent",
            cursor: "pointer",
            textAlign: "left",
            transition: "color 0.12s, background 0.12s",
            lineHeight: "22px",
            minHeight: 26,
          }}
        >
          <IndentGuides depth={depth} />
          <span style={{ color: "var(--text-faint)", display: "flex" }}>
            <IconChevron expanded={expanded} />
          </span>
          <span style={{ color: isUnsorted ? "var(--accent-text)" : "var(--text-faint)", display: "flex", flexShrink: 0 }}>
            {isUnsorted ? <IconGeneral /> : (expanded ? <IconFolderOpen /> : <IconFolderClosed />)}
          </span>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
            {displayName}
          </span>
          {folder.document_count > 0 && (
            <span style={{
              flexShrink: 0,
              fontSize: 10,
              color: "var(--text-faint)",
              letterSpacing: "0.3px",
              fontVariantNumeric: "tabular-nums",
              fontWeight: 400,
            }}>
              {folder.document_count}
            </span>
          )}
        </button>
      )}

      {expanded && (
        <div>
          {/* Sub-folders first */}
          {folder.children.map((child) => (
            <TreeFolder
              key={child.path}
              folder={child}
              depth={depth + 1}
              expanded={expandedSet.has(child.path)}
              onToggle={() => onToggleFolder(child.path)}
              activeDocId={activeDocId}
              onFileClick={onFileClick}
              expandedSet={expandedSet}
              onToggleFolder={onToggleFolder}
              activeFolder={activeFolder}
              onFolderClick={onFolderClick}
              onContextMenu={onContextMenu}
              renameTarget={renameTarget}
              onRenameSubmit={onRenameSubmit}
              onRenameCancel={onRenameCancel}
            />
          ))}
          {/* Files */}
          {folder.documents.map((doc) => (
            <TreeFile
              key={doc.id}
              doc={doc}
              depth={depth + 1}
              isActive={!activeFolder && activeDocId === doc.id}
              onClick={() => onFileClick(doc.id)}
              onContextMenu={onContextMenu}
              isRenaming={renameTarget?.type === 'file' && renameTarget.id === doc.id}
              onRenameSubmit={onRenameSubmit}
              onRenameCancel={onRenameCancel}
            />
          ))}
          {/* Empty folder hint */}
          {folder.children.length === 0 && folder.documents.length === 0 && (
            <div style={{
              paddingLeft: 8 + (depth + 1) * INDENT_PX,
              fontSize: 12,
              color: "var(--text-faint)",
              fontStyle: "italic",
              lineHeight: "24px",
              opacity: 0.7,
            }}>
              empty
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── FileTree (main export) ────────────────────────────────────────

export default function FileTree({ tree, activeDocId, onFileClick, activeFolder, onFolderClick, triggerRefreshFolders, onUploadToFolder, onCrawlToFolder }: FileTreeProps) {
  const [expandedSet, setExpandedSet] = useState<Set<string>>(loadExpanded);
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [renameTarget, setRenameTarget] = useState<{ id: string, type: 'file' | 'folder' } | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ContextMenuState['target'] | null>(null);
  const navigate = useNavigate();

  // Persist expand state
  useEffect(() => {
    saveExpanded(expandedSet);
  }, [expandedSet]);

  useEffect(() => {
    const handleGlobalClick = () => setContextMenu(null);
    window.addEventListener("click", handleGlobalClick);
    return () => window.removeEventListener("click", handleGlobalClick);
  }, []);

  const toggleFolder = useCallback((path: string) => {
    setExpandedSet((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const handleContextMenu = useCallback((e: React.MouseEvent, target: ContextMenuState['target']) => {
    setContextMenu({ x: e.clientX, y: e.clientY, target });
  }, []);

  const handleRenameSubmit = async (id: string, newName: string, type: 'file' | 'folder') => {
    setRenameTarget(null);
    if (!newName.trim()) return;
    try {
      if (type === 'folder') {
        const parts = id.split('/');
        parts.pop();
        const parentPath = parts.join('/');
        const newPath = parentPath ? `${parentPath}/${newName}` : newName;
        if (newPath !== id) {
          await apiPut('/api/folders/rename', { path: id, new_path: newPath });
        }
      } else {
        await apiPut(`/api/documents/${id}`, { title: newName, filename: newName });
      }
      if (triggerRefreshFolders) triggerRefreshFolders();
    } catch (err) {
      alert("Failed to rename: " + (err as Error).message);
    }
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    try {
      if (deleteTarget.type === 'folder') {
        const encodedPath = deleteTarget.path.split('/').map(encodeURIComponent).join('/');
        await apiDelete(`/api/folders/${encodedPath}`);
      } else {
        await apiDelete(`/api/documents/${encodeURIComponent(deleteTarget.id)}`);
        if (deleteTarget.id === activeDocId) {
          navigate("/");
        }
      }
      if (triggerRefreshFolders) triggerRefreshFolders();
    } catch (err) {
      alert("Failed to delete: " + (err as Error).message);
    } finally {
      setDeleteTarget(null);
    }
  };

  const renderOverlays = () => (
    <>
      {contextMenu && (
        <div
          className="context-menu"
          style={{
            position: "fixed",
            left: contextMenu.x,
            top: contextMenu.y,
            zIndex: 1000,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {contextMenu.target.type === 'folder' && onUploadToFolder && (
            <button
              className="context-menu-item"
              onClick={(e) => {
                e.preventDefault();
                onUploadToFolder(contextMenu.target.path);
                setContextMenu(null);
              }}
            >
              Upload File
            </button>
          )}
          {contextMenu.target.type === 'folder' && onCrawlToFolder && (
            <button
              className="context-menu-item"
              onClick={(e) => {
                e.preventDefault();
                onCrawlToFolder(contextMenu.target.path);
                setContextMenu(null);
              }}
            >
              Crawl URL
            </button>
          )}
          {contextMenu.target.type === 'folder' && (onUploadToFolder || onCrawlToFolder) && (
            <div className="context-menu-separator" />
          )}
          {contextMenu.target.path !== UNSORTED_FOLDER && (
            <>
          <button
            className="context-menu-item"
            onClick={(e) => {
              e.preventDefault();
              setRenameTarget({ id: contextMenu.target.id, type: contextMenu.target.type });
              setContextMenu(null);
            }}
          >
            Rename
          </button>
          <button
            className="context-menu-item context-menu-item-danger"
            onClick={(e) => {
              e.preventDefault();
              setDeleteTarget(contextMenu.target);
              setContextMenu(null);
            }}
          >
            Delete
          </button>
            </>
          )}
        </div>
      )}

      {deleteTarget && (
        <div className="dialog-overlay" onClick={(e) => { if (e.target === e.currentTarget) setDeleteTarget(null); }}>
          <div className="dialog-panel" style={{ maxWidth: 400 }}>
            <h2 style={{ fontSize: 18, fontWeight: 500, color: "var(--text-primary)", margin: "0 0 16px 0" }}>
              Delete {deleteTarget.type === 'folder' ? 'Folder' : 'Document'}
            </h2>
            <p style={{ fontSize: 14, color: "var(--text-secondary)", marginBottom: 24, lineHeight: 1.5 }}>
              Are you sure you want to delete {deleteTarget.type === 'folder' ? 'the folder' : 'the document'} <strong>"{deleteTarget.name}"</strong>?
              {deleteTarget.type === 'folder' && " All contents inside this folder will also be deleted. This action cannot be undone."}
            </p>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
              <button onClick={(e) => { e.preventDefault(); setDeleteTarget(null); }} className="btn-ghost">Cancel</button>
              <button onClick={(e) => { e.preventDefault(); confirmDelete(); }} className="btn-primary" style={{ background: "var(--error)", borderColor: "var(--error)" }}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );

  if (tree.length === 0) {
    return (
      <div style={{ position: "relative" }}>
        <div style={{
          padding: "16px",
          textAlign: "center",
          fontSize: 12,
          color: "var(--text-faint)",
          fontStyle: "italic",
        }}>
          No folders yet
        </div>
        {renderOverlays()}
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        {tree.map((folder) => (
          <TreeFolder
            key={folder.path}
            folder={folder}
            depth={0}
            expanded={expandedSet.has(folder.path)}
            onToggle={() => toggleFolder(folder.path)}
            activeDocId={activeDocId}
            onFileClick={onFileClick}
            expandedSet={expandedSet}
            onToggleFolder={toggleFolder}
            activeFolder={activeFolder}
            onFolderClick={onFolderClick}
            onContextMenu={handleContextMenu}
            renameTarget={renameTarget}
            onRenameSubmit={handleRenameSubmit}
            onRenameCancel={() => setRenameTarget(null)}
          />
        ))}
      </div>

      {renderOverlays()}
    </div>
  );
}
