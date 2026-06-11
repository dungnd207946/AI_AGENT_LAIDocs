import { useState } from "react";

const IconBranch = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="6" cy="6" r="3" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="12" r="3" />
    <path d="M6 9v6" /><path d="M9 18h3a3 3 0 0 0 3-3v-1.5" /><path d="M9 6h3a3 3 0 0 1 3 3v1.5" />
  </svg>
);

const IconArrow = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12" /><polyline points="13 6 19 12 13 18" />
  </svg>
);

const IconChevron = ({ open }: { open: boolean }) => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}>
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

interface Hop {
  rel: string;
  target: string;
}
interface ParsedPath {
  start: string;
  hops: Hop[];
}

/** Parse render_reasoning() output into structured paths.
 *  Each numbered line looks like:
 *    "1. Marco Ruiz --[reports to]--> Lena Hoffmann --[born in]--> Lyon"
 */
function parseChain(chain: string): ParsedPath[] {
  const edgeRe = /--\[(.*?)\]-->/g;
  const paths: ParsedPath[] = [];
  for (const raw of chain.split("\n")) {
    const line = raw.replace(/^\s*\d+\.\s*/, "").trim();
    if (!line || !line.includes("--[")) continue;
    const nodes = line.split(/--\[.*?\]-->/).map((n) => n.trim());
    const rels: string[] = [];
    let m: RegExpExecArray | null;
    edgeRe.lastIndex = 0;
    while ((m = edgeRe.exec(line)) !== null) rels.push(m[1].trim());
    if (nodes.length < 2 || rels.length === 0) continue;
    const hops: Hop[] = rels.map((rel, i) => ({ rel, target: nodes[i + 1] ?? "" }));
    paths.push({ start: nodes[0], hops });
  }
  return paths;
}

function NodeChip({ label, accent }: { label: string; accent?: boolean }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium whitespace-nowrap ${
        accent
          ? "bg-[var(--accent-subtle)] text-[var(--accent-text)] border border-[var(--border-glow)]"
          : "bg-[var(--surface-alt)] text-[var(--text-primary)] border border-[var(--border-strong)]"
      }`}
    >
      {label}
    </span>
  );
}

export default function ReasoningChain({ chain }: { chain: string }) {
  const [open, setOpen] = useState(true);
  const paths = parseChain(chain);
  if (paths.length === 0) return null;

  return (
    <div className="mt-2 rounded-lg border border-[var(--border)] bg-[var(--surface-alt)]/40 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium uppercase tracking-wider text-[var(--text-faint)] hover:text-[var(--text-muted)] transition-colors"
      >
        <span className="text-[var(--accent-text)] flex"><IconBranch /></span>
        <span>Reasoning path</span>
        <span className="text-[var(--text-faint)] normal-case tracking-normal">
          · {paths.length} chain{paths.length > 1 ? "s" : ""}
        </span>
        <span className="ml-auto flex"><IconChevron open={open} /></span>
      </button>

      {open && (
        <div className="px-2.5 pb-2.5 flex flex-col gap-2">
          {paths.map((p, idx) => (
            <div key={idx} className="flex flex-wrap items-center gap-1.5">
              <NodeChip label={p.start} accent />
              {p.hops.map((hop, i) => (
                <span key={i} className="flex items-center gap-1.5">
                  <span className="inline-flex items-center gap-1 text-[var(--text-faint)]">
                    <IconArrow />
                    <span className="text-[10px] italic text-[var(--text-muted)] whitespace-nowrap">{hop.rel}</span>
                    <IconArrow />
                  </span>
                  <NodeChip label={hop.target} accent={i === p.hops.length - 1} />
                </span>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
