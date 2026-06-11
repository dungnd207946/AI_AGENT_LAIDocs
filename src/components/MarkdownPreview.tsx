import ReactMarkdown from "react-markdown";
import { API_BASE } from "../lib/sidecar";
import remarkGfm from "remark-gfm";

interface MarkdownPreviewProps {
  content: string;
  compact?: boolean;
}

export default function MarkdownPreview({ content, compact }: MarkdownPreviewProps) {
  const fs = compact ? 12 : 15;
  const headingFont = compact ? "inherit" : "'Outfit', ui-sans-serif, system-ui, sans-serif";
  return (
    <div style={{ overflow: "auto", height: compact ? "auto" : "100%", padding: compact ? 0 : "28px 32px" }}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 style={{ fontFamily: headingFont, fontSize: compact ? fs + 2 : 24, fontWeight: compact ? 600 : 600, color: "var(--text-primary)", marginBottom: compact ? 8 : 16, marginTop: compact ? 12 : 24, paddingBottom: compact ? 0 : 8, borderBottom: compact ? "none" : "1px solid var(--border)", lineHeight: 1.3, letterSpacing: "-0.3px" }}>
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 style={{ fontFamily: headingFont, fontSize: compact ? fs + 1 : 20, fontWeight: compact ? 600 : 600, color: "var(--text-primary)", marginBottom: compact ? 6 : 12, marginTop: compact ? 10 : 20, paddingBottom: compact ? 0 : 6, borderBottom: compact ? "none" : "1px solid var(--border)", lineHeight: 1.3, letterSpacing: "-0.2px" }}>
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 style={{ fontSize: compact ? fs : 17, fontWeight: 500, color: "var(--text-primary)", marginBottom: compact ? 4 : 10, marginTop: compact ? 8 : 18, lineHeight: 1.4 }}>
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 style={{ fontSize: compact ? fs : 15, fontWeight: 500, color: "var(--text-primary)", marginBottom: compact ? 4 : 8, marginTop: compact ? 6 : 14, lineHeight: 1.4 }}>
              {children}
            </h4>
          ),
          p: ({ children }) => (
            <p style={{ color: "var(--text-secondary)", marginBottom: compact ? 8 : 16, lineHeight: compact ? 1.6 : 1.75, fontSize: fs }}>
              {children}
            </p>
          ),
          a: ({ href, children }) => (
            <a href={href} style={{ color: "var(--accent-text)", textDecoration: "underline", textUnderlineOffset: 3, transition: "opacity 0.15s" }} target="_blank" rel="noopener noreferrer" download={href?.endsWith(".md")}>
              {children}
            </a>
          ),
          ul: ({ children }) => (
            <ul style={{ color: "var(--text-secondary)", marginBottom: compact ? 8 : 16, paddingLeft: compact ? 16 : 20, listStyleType: "disc", fontSize: fs, lineHeight: compact ? 1.6 : 1.7 }}>
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol style={{ color: "var(--text-secondary)", marginBottom: compact ? 8 : 16, paddingLeft: compact ? 16 : 20, listStyleType: "decimal", fontSize: fs, lineHeight: compact ? 1.6 : 1.7 }}>
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li style={{ marginBottom: 4, lineHeight: 1.7 }}>{children}</li>
          ),
          blockquote: ({ children }) => (
            <blockquote style={{
              borderLeft: "3px solid var(--accent)",
              paddingLeft: 16, margin: "16px 0",
              color: "var(--text-muted)", fontStyle: "italic",
              background: "var(--surface-alt)", borderRadius: "0 8px 8px 0",
              padding: "12px 18px",
            }}>
              {children}
            </blockquote>
          ),
          code: ({ className, children }) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code style={{
                  background: "var(--surface-alt)", color: "var(--text-secondary)",
                  padding: "2px 6px", borderRadius: 5, fontSize: 13,
                  fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Courier New', monospace",
                  border: "1px solid var(--border)",
                }}>
                  {children}
                </code>
              );
            }
            return (
              <code style={{ fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Courier New', monospace", fontSize: 13 }} className={className}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre style={{
              background: "var(--surface-alt)", borderRadius: 10, padding: "16px 20px",
              marginBottom: 16, overflowX: "auto",
              border: "1px solid var(--border)", fontSize: 13,
              fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Courier New', monospace",
            }}>
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div style={{ overflowX: "auto", marginBottom: 16 }}>
              <table style={{ minWidth: "100%", borderCollapse: "collapse", border: "1px solid var(--border)", borderRadius: 10 }}>
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead style={{ background: "var(--surface-alt)" }}>{children}</thead>
          ),
          th: ({ children }) => (
            <th style={{ border: "1px solid var(--border)", padding: "9px 14px", textAlign: "left", fontSize: 12, fontWeight: 500, color: "var(--text-muted)", letterSpacing: "1px", textTransform: "uppercase" }}>
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td style={{ border: "1px solid var(--border)", padding: "8px 14px", fontSize: 14, color: "var(--text-secondary)" }}>
              {children}
            </td>
          ),
          hr: () => <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "28px 0" }} />,
          img: ({ src, alt }) => {
            // Rewrite vault asset URLs to point at the backend server.
            // Stored markdown uses relative paths like /assets/xxx.png which
            // the browser would resolve against the frontend origin (Vite/Tauri)
            // instead of the FastAPI sidecar at API_BASE.
            const resolvedSrc = src?.startsWith("/assets/") ? `${API_BASE}${src}` : src;
            return (
              <img src={resolvedSrc} alt={alt ?? ""} style={{ maxWidth: "100%", borderRadius: 10, margin: "12px 0", border: "1px solid var(--border)" }} />
            );
          },
          strong: ({ children }) => (
            <strong style={{ fontWeight: 500, color: "var(--text-primary)" }}>{children}</strong>
          ),
          em: ({ children }) => (
            <em style={{ fontStyle: "italic", color: "var(--text-secondary)" }}>{children}</em>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
