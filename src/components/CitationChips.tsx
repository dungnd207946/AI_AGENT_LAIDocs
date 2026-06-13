import { useState } from "react";
import type { Evidence } from "../lib/sidecar";

const IconDoc = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>
);

const IconTable = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <line x1="3" y1="9" x2="21" y2="9" /><line x1="3" y1="15" x2="21" y2="15" />
    <line x1="12" y1="3" x2="12" y2="21" />
  </svg>
);

const IconImage = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
  </svg>
);

const IconShield = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <polyline points="9 12 11 14 15 10" />
  </svg>
);

function kindIcon(kind: string) {
  if (kind === "table") return <IconTable />;
  if (kind === "image") return <IconImage />;
  return <IconDoc />;
}

function chipLabel(ev: Evidence): string {
  if (ev.heading_path && ev.heading_path.length) {
    return ev.heading_path[ev.heading_path.length - 1];
  }
  return ev.title || ev.unit_id;
}

interface CitationChipsProps {
  evidence: Evidence[];
  onJump?: (ev: Evidence) => void;
}

export default function CitationChips({ evidence, onJump }: CitationChipsProps) {
  const [hovered, setHovered] = useState<string | null>(null);
  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="mt-2 flex flex-col gap-1.5">
      {/* Grounding badge */}
      <div className="flex items-center gap-1.5">
        <span
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium text-[var(--success)] bg-[var(--success-bg,rgba(74,222,128,0.1))] border border-[rgba(74,222,128,0.2)]"
          title="Every claim is grounded in the document sections below"
        >
          <IconShield />
          Grounded · {evidence.length} source{evidence.length > 1 ? "s" : ""}
        </span>
      </div>

      {/* Citation chips */}
      <div className="flex flex-wrap gap-1.5">
        {evidence.map((ev) => (
          <div key={ev.unit_id} className="relative">
            <button
              type="button"
              onClick={() => onJump?.(ev)}
              onMouseEnter={() => setHovered(ev.unit_id)}
              onMouseLeave={() => setHovered((h) => (h === ev.unit_id ? null : h))}
              title="Jump to this section in the document"
              className="group inline-flex items-center gap-1.5 max-w-[220px] px-2 py-1 rounded-md text-[11px] text-[var(--text-muted)] bg-[var(--surface-alt)] border border-[var(--border)] hover:border-[var(--accent)] hover:text-[var(--accent-text)] transition-all cursor-pointer"
            >
              <span className="shrink-0 text-[var(--text-faint)] group-hover:text-[var(--accent-text)] flex">
                {kindIcon(ev.kind)}
              </span>
              <span className="truncate">{chipLabel(ev)}</span>
            </button>

            {hovered === ev.unit_id && ev.preview && (
              <div className="absolute z-40 left-0 top-[calc(100%+4px)] w-72 max-w-[80vw] p-2.5 rounded-lg border border-[var(--border-strong)] bg-[var(--surface)] shadow-2xl shadow-black/40 scale-in origin-top-left pointer-events-none">
                {ev.heading_path && ev.heading_path.length > 0 && (
                  <div className="text-[10px] text-[var(--text-faint)] mb-1 truncate">
                    {ev.heading_path.join(" › ")}
                  </div>
                )}
                <div className="text-[11px] leading-relaxed text-[var(--text-secondary)] line-clamp-4">
                  {ev.preview}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
