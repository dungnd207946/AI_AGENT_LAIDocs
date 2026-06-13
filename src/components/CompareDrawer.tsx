import type { CompareResult, CompareArm, Evidence } from "../lib/sidecar";
import MarkdownPreview from "./MarkdownPreview";

const IconX = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const IconScale = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v18" /><path d="M5 7h14" /><path d="M5 7l-3 6a4 4 0 0 0 6 0z" /><path d="M19 7l3 6a4 4 0 0 1-6 0z" />
    <path d="M8 21h8" />
  </svg>
);

const IconSpark = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" />
  </svg>
);

function unitLabel(ev: Evidence): string {
  if (ev.heading_path && ev.heading_path.length) return ev.heading_path[ev.heading_path.length - 1];
  return ev.title || ev.unit_id;
}

function ArmColumn({
  arm,
  title,
  accent,
  bridgeIds,
  badge,
}: {
  arm: CompareArm;
  title: string;
  accent: string;
  bridgeIds: Set<string>;
  badge?: string;
}) {
  return (
    <div className="flex-1 min-w-0 flex flex-col rounded-xl border border-[var(--border-strong)] bg-[var(--surface)] overflow-hidden">
      <div className="flex items-center gap-2 px-3.5 py-2.5 border-b border-[var(--border)]" style={{ background: "var(--surface-alt)" }}>
        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: accent, boxShadow: `0 0 8px ${accent}` }} />
        <span className="text-[12px] font-semibold text-[var(--text-primary)]">{title}</span>
        {badge && (
          <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded-full font-medium" style={{ color: accent, background: "var(--accent-subtle)" }}>
            {badge}
          </span>
        )}
      </div>

      {/* Answer */}
      <div className="px-3.5 py-3 text-[12px] text-[var(--text-secondary)] border-b border-[var(--border)]">
        <MarkdownPreview content={arm.answer} compact />
      </div>

      {/* Retrieved units */}
      <div className="px-3.5 py-2.5">
        <div className="text-[10px] uppercase tracking-wider text-[var(--text-faint)] mb-1.5">
          Retrieved units · {arm.units.length}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {arm.units.map((u) => {
            const isBridge = bridgeIds.has(u.unit_id);
            return (
              <span
                key={u.unit_id}
                title={isBridge ? "Bridge passage — only GraphRAG recovered this" : u.preview}
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] max-w-[200px] ${
                  isBridge
                    ? "text-[var(--success)] bg-[var(--success-bg,rgba(74,222,128,0.12))] border border-[rgba(74,222,128,0.35)] font-medium"
                    : "text-[var(--text-muted)] bg-[var(--surface-alt)] border border-[var(--border)]"
                }`}
              >
                {isBridge && <span className="flex shrink-0"><IconSpark /></span>}
                <span className="truncate">{unitLabel(u)}</span>
              </span>
            );
          })}
          {arm.units.length === 0 && (
            <span className="text-[11px] text-[var(--text-faint)] italic">no units retrieved</span>
          )}
        </div>
      </div>
    </div>
  );
}

interface CompareDrawerProps {
  question: string;
  loading: boolean;
  error: string | null;
  result: CompareResult | null;
  onClose: () => void;
}

export default function CompareDrawer({ question, loading, error, result, onClose }: CompareDrawerProps) {
  const bridgeIds = new Set(result?.bridge_unit_ids ?? []);

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center p-4 fade-in"
      style={{ background: "rgba(0,0,0,0.55)", backdropFilter: "blur(2px)" }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl max-h-full flex flex-col rounded-2xl border border-[var(--border-strong)] bg-[var(--surface)] shadow-2xl shadow-black/50 scale-in overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-2.5 px-4 py-3 border-b border-[var(--border)] shrink-0">
          <span className="text-[var(--accent-text)] flex"><IconScale /></span>
          <div className="flex flex-col min-w-0">
            <span className="text-[12px] font-semibold text-[var(--text-primary)]">Plain RAG vs GraphRAG</span>
            <span className="text-[11px] text-[var(--text-faint)] truncate">{question}</span>
          </div>
          <button onClick={onClose} className="btn-icon ml-auto shrink-0" title="Close"><IconX /></button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading && (
            <div className="flex flex-col items-center justify-center py-16 gap-3">
              <div className="w-6 h-6 border-2 border-[var(--border)] border-t-[var(--accent)] rounded-full spin" />
              <p className="text-[12px] text-[var(--text-muted)] m-0">Running both retrievers on the same question…</p>
            </div>
          )}

          {error && !loading && (
            <div className="p-3 rounded-lg border border-[rgba(248,113,113,0.2)] bg-[var(--error-bg)] text-[12px] text-[var(--error)]">
              {error}
            </div>
          )}

          {result && !loading && !error && (
            <div className="flex flex-col gap-3">
              <div className="flex flex-col md:flex-row gap-3">
                <ArmColumn arm={result.rag} title="Plain RAG" accent="var(--text-muted)" bridgeIds={bridgeIds} />
                <ArmColumn arm={result.graph} title="GraphRAG" accent="var(--accent)" bridgeIds={bridgeIds} badge="graph walk" />
              </div>

              <div className="text-[11px] text-[var(--text-muted)] px-1 leading-relaxed">
                {bridgeIds.size > 0 ? (
                  <>
                    <span className="text-[var(--success)] font-medium">{bridgeIds.size} bridge passage{bridgeIds.size > 1 ? "s" : ""}</span>{" "}
                    surfaced only by walking the entity-relation graph (highlighted above). Same model, same
                    question — the only variable is whether the graph was walked.
                  </>
                ) : (
                  <>Both retrievers returned the same units for this question — try a multi-hop question whose answer is split across sections (e.g. <em>“Where was the founder born?”</em>).</>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
