import { useEffect, useState } from "react";
import { apiGet } from "../lib/sidecar";
import { useUpload } from "../context/UploadContext";

interface Folder { path: string; name: string; document_count: number; }
interface CrawlDialogProps { open: boolean; onClose: () => void; onCrawlSuccess: () => void; initialFolder?: string | null; }

const IconX = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

const IconGlobe = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <line x1="2" y1="12" x2="22" y2="12"/>
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
  </svg>
);

export default function CrawlDialog({ open, onClose, onCrawlSuccess, initialFolder }: CrawlDialogProps) {
  const [folders, setFolders] = useState<Folder[]>([]);
  const [selectedFolder, setSelectedFolder] = useState("");
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");

  const { startCrawl } = useUpload();

  useEffect(() => {
    if (!open) return;
    apiGet<Folder[]>("/api/folders/").then(setFolders).catch(() => setFolders([]));
    setUrl(""); setSelectedFolder(initialFolder || ""); setError("");
  }, [open, initialFolder]);

  if (!open) return null;

  const handleCrawl = () => {
    const trimmedUrl = url.trim();
    if (!trimmedUrl) { setError("Please enter a URL"); return; }
    try { new URL(trimmedUrl); } catch { setError("Please enter a valid URL (e.g. https://example.com)"); return; }
    setError("");
    // Start streaming crawl (non-blocking) then close dialog immediately
    startCrawl(trimmedUrl, selectedFolder || "unsorted");
    onCrawlSuccess();
    onClose();
  };

  return (
    <div className="dialog-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="dialog-panel">
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
          <h2 style={{ fontSize: 18, fontWeight: 500, color: "var(--text-primary)", margin: 0 }}>Crawl URL</h2>
          <button onClick={onClose} className="btn-icon"><IconX /></button>
        </div>

        {error && (
          <div style={{ marginBottom: 16, padding: "10px 14px", background: "rgba(192,112,112,0.1)", border: "1px solid rgba(192,112,112,0.3)", borderRadius: 8, fontSize: 13, color: "var(--error)" }}>
            {error}
          </div>
        )}

        {/* URL input */}
        <div style={{ marginBottom: 20 }}>
          <label style={{ display: "block", fontSize: 11, color: "var(--text-muted)", letterSpacing: "1.4px", textTransform: "uppercase", marginBottom: 6 }}>URL</label>
          <div style={{ position: "relative" }}>
            <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }}>
              <IconGlobe />
            </span>
            <input
              type="text"
              value={url}
              onChange={(e) => { setUrl(e.target.value); setError(""); }}
              onKeyDown={(e) => { if (e.key === "Enter") handleCrawl(); }}
              placeholder="https://example.com"
              autoFocus
              className="warp-input"
              style={{ paddingLeft: 36 }}
            />
          </div>
        </div>

        {/* Folder select */}
        <div style={{ marginBottom: 24 }}>
          <label style={{ display: "block", fontSize: 11, color: "var(--text-muted)", letterSpacing: "1.4px", textTransform: "uppercase", marginBottom: 6 }}>Folder</label>
          <select value={selectedFolder} onChange={(e) => setSelectedFolder(e.target.value)} className="warp-input" style={{ appearance: "none" }}>
            <option value="">None</option>
            {folders.map((f) => <option key={f.path} value={f.path}>{f.path === "unsorted" ? "General" : (f.name || f.path)}</option>)}
          </select>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
          <button onClick={onClose} className="btn-ghost">Cancel</button>
          <button onClick={handleCrawl} className="btn-primary">
            Crawl
          </button>
        </div>
      </div>
    </div>
  );
}
