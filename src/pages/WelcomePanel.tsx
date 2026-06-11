import { useState } from "react";
import UploadDialog from "../components/UploadDialog";
import CrawlDialog from "../components/CrawlDialog";

// ── SVG Icons ─────────────────────────────────────────────────────

const IconUpload = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const IconGlobe = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="2" y1="12" x2="22" y2="12" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
  </svg>
);

// ── Technology badge ──────────────────────────────────────────────

function TechBadge({ label }: { label: string }) {
  return (
    <span style={{
      display: "inline-flex",
      alignItems: "center",
      fontSize: 10,
      fontWeight: 500,
      color: "var(--text-muted)",
      background: "var(--surface-alt)",
      border: "1px solid var(--border)",
      borderRadius: 5,
      padding: "2px 8px",
      letterSpacing: "0.3px",
      whiteSpace: "nowrap",
    }}>
      {label}
    </span>
  );
}

// ── Technology showcase card ──────────────────────────────────────

interface TechCardProps {
  icon: React.ReactNode;
  label: string;
  headline: string;
  description: string;
  techStack: string[];
  accentColor: string;
  delay: number;
}

function TechCard({ icon, label, headline, description, techStack, accentColor, delay }: TechCardProps) {
  return (
    <div
      className="fade-in-up"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 12,
        padding: "22px 20px 18px",
        borderRadius: 14,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        animationDelay: `${delay}ms`,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Subtle top accent line */}
      <div style={{
        position: "absolute",
        top: 0,
        left: "20%",
        right: "20%",
        height: 1,
        background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)`,
        opacity: 0.5,
      }} />

      {/* Category label */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}>
        <div style={{
          width: 28, height: 28, borderRadius: 8,
          background: `${accentColor}15`,
          border: `1px solid ${accentColor}25`,
          display: "flex", alignItems: "center", justifyContent: "center",
          color: accentColor,
          flexShrink: 0,
        }}>
          {icon}
        </div>
        <span style={{
          fontSize: 9,
          fontWeight: 600,
          letterSpacing: "1.5px",
          textTransform: "uppercase" as const,
          color: accentColor,
          opacity: 0.85,
        }}>
          {label}
        </span>
      </div>

      {/* Headline */}
      <div style={{
        fontSize: 14,
        fontWeight: 600,
        color: "var(--text-primary)",
        lineHeight: 1.35,
        letterSpacing: "-0.1px",
      }}>
        {headline}
      </div>

      {/* Description */}
      <div style={{
        fontSize: 12,
        color: "var(--text-muted)",
        lineHeight: 1.6,
      }}>
        {description}
      </div>

      {/* Tech stack badges */}
      <div style={{
        display: "flex",
        flexWrap: "wrap" as const,
        gap: 5,
        marginTop: 2,
      }}>
        {techStack.map((tech) => (
          <TechBadge key={tech} label={tech} />
        ))}
      </div>
    </div>
  );
}

// ── Card icon components ──────────────────────────────────────────

const IconDocConvert = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </svg>
);

const IconWeb = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="2" y1="12" x2="22" y2="12" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
  </svg>
);

const IconMessageAI = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    <circle cx="9" cy="10" r="1" fill="currentColor" />
    <circle cx="15" cy="10" r="1" fill="currentColor" />
  </svg>
);

// ── WelcomePanel ──────────────────────────────────────────────────

export default function WelcomePanel() {
  const [showUpload, setShowUpload] = useState(false);
  const [showCrawl, setShowCrawl] = useState(false);

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 40,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Ambient background glow */}
      <div style={{
        position: "absolute",
        top: "-20%",
        left: "30%",
        width: 500,
        height: 500,
        borderRadius: "50%",
        background: "radial-gradient(circle, var(--accent-subtle) 0%, transparent 70%)",
        pointerEvents: "none",
        opacity: 0.7,
      }} />
      <div style={{
        position: "absolute",
        bottom: "-30%",
        right: "20%",
        width: 400,
        height: 400,
        borderRadius: "50%",
        background: "radial-gradient(circle, rgba(99, 102, 241, 0.04) 0%, transparent 70%)",
        pointerEvents: "none",
      }} />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
          maxWidth: 620,
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Hero icon */}
        <div
          className="fade-in float"
          style={{
            width: 72,
            height: 72,
            borderRadius: 20,
            background: "linear-gradient(135deg, var(--accent-subtle), var(--surface-alt))",
            border: "1px solid var(--border-glow)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: 28,
            boxShadow: "0 8px 30px var(--accent-subtle)",
          }}
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-text)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
          </svg>
        </div>

        <h1
          className="fade-in-up heading-display"
          style={{
            margin: "0 0 12px",
            animationDelay: "0.06s",
          }}
        >
          Welcome to LAIDocs
        </h1>

        <p
          className="fade-in-up"
          style={{
            fontSize: 14,
            color: "var(--text-muted)",
            lineHeight: 1.65,
            margin: "0 0 32px",
            maxWidth: 420,
            animationDelay: "0.12s",
          }}
        >
          Your intelligent knowledge base. Upload documents, crawl web pages, and chat with your content using AI.
        </p>

        {/* Action Buttons */}
        <div
          className="fade-in-up"
          style={{ display: "flex", gap: 12, marginBottom: 40, animationDelay: "0.18s" }}
        >
          <button
            onClick={() => setShowUpload(true)}
            className="btn-accent"
          >
            <IconUpload />
            Upload File
          </button>

          <button
            onClick={() => setShowCrawl(true)}
            className="btn-ghost"
          >
            <IconGlobe />
            Crawl URL
          </button>
        </div>

        {/* Section divider with label */}
        <div
          className="fade-in-up"
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            width: "100%",
            marginBottom: 20,
            animationDelay: "0.22s",
          }}
        >
          <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
          <span style={{
            fontSize: 9,
            fontWeight: 600,
            letterSpacing: "2px",
            textTransform: "uppercase" as const,
            color: "var(--text-faint)",
          }}>
            Powered by
          </span>
          <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
        </div>

        {/* Technology showcase cards */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 12,
          width: "100%",
          textAlign: "left",
        }}>
          <TechCard
            icon={<IconDocConvert />}
            label="Document Engine"
            headline="Smart Document Conversion"
            description="Automatically extract text, layouts, and tables from complex files like PDF, XLSX, DOCX, PPTX into clean Markdown."
            techStack={["Docling", "Markitdown"]}
            accentColor="#34d399"
            delay={260}
          />
          <TechCard
            icon={<IconWeb />}
            label="Web Engine"
            headline="Intelligent Web Crawling"
            description="Extract webpage content into readable Markdown, intelligently stripping away ads and unnecessary clutter."
            techStack={["Crawl4AI"]}
            accentColor="#a5b4fc"
            delay={320}
          />
          <TechCard
            icon={<IconMessageAI />}
            label="Chat Engine"
            headline="Agentic Chat with Documents"
            description="Engage with a DeepAgents-powered assistant that strictly answers from context, remembers history, and manages sessions."
            techStack={["DeepAgents", "Reasoning-based RAG"]}
            accentColor="#fbbf24"
            delay={380}
          />
        </div>

        {/* Copyright & Version */}
        <div
          className="fade-in-up"
          style={{
            marginTop: 32,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 6,
            animationDelay: "0.44s",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 11, color: "var(--text-faint)", letterSpacing: "0.2px" }}>
              © 2026 Dino
            </span>
            <span style={{ fontSize: 11, color: "var(--text-faint)", opacity: 0.4 }}>·</span>
            <span style={{
              fontSize: 10,
              fontWeight: 500,
              color: "var(--text-faint)",
              background: "var(--surface-alt)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              padding: "1px 6px",
              letterSpacing: "0.3px",
            }}>
              v1.0
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--text-faint)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="4" width="20" height="16" rx="2" />
              <path d="M22 4L12 13 2 4" />
            </svg>
            <span style={{ fontSize: 11, color: "var(--text-faint)", letterSpacing: "0.2px" }}>
              Chatbot_GENAI_MASTER_PROJECT
            </span>
          </div>
        </div>
      </div>

      <UploadDialog
        open={showUpload}
        onClose={() => setShowUpload(false)}
        initialFolder="unsorted"
        onUploadSuccess={() => setShowUpload(false)}
      />
      <CrawlDialog
        open={showCrawl}
        onClose={() => setShowCrawl(false)}
        initialFolder="unsorted"
        onCrawlSuccess={() => setShowCrawl(false)}
      />
    </div>
  );
}
