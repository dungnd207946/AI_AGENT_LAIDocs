import { useEffect, useState, useCallback } from "react";
import { apiGet, apiPost, apiPut } from "../lib/sidecar";
import DataTab, { type BackupStats, type PreviewResult } from "../components/DataTab";

interface ServiceConfig { base_url: string; api_key: string; model: string; }
interface RerankerConfig extends ServiceConfig {
  enabled: boolean;
  top_n: number;
  candidate_k: number;
  timeout_s: number;
}
interface SettingsData { llm: ServiceConfig; reranker: RerankerConfig; port: number; }
interface TestResult { type: "success" | "error"; message: string; }

// ── Field label ───────────────────────────────────────────────────
function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label style={{
      display: "block", fontSize: 10, color: "var(--text-faint)",
      letterSpacing: "1.4px", textTransform: "uppercase", marginBottom: 7, fontWeight: 500,
    }}>
      {children}
    </label>
  );
}

// ── Text input ────────────────────────────────────────────────────
function WarpInput({ label, value, onChange, placeholder, type = "text" }: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string; type?: string;
}) {
  const [visible, setVisible] = useState(false);
  const isPassword = type === "password";
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <div style={{ position: "relative" }}>
        <input
          type={isPassword && !visible ? "password" : "text"}
          value={value}
          onChange={(e) => onChange(e.target.value.trim())}
          placeholder={placeholder}
          className="warp-input"
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setVisible(!visible)}
            style={{
              position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)",
              background: "none", border: "none", cursor: "pointer",
              fontSize: 10, color: "var(--text-faint)", letterSpacing: "0.5px",
              transition: "color 0.15s", padding: "2px 4px",
            }}
            tabIndex={-1}
            onMouseEnter={e => (e.currentTarget.style.color = "var(--text-muted)")}
            onMouseLeave={e => (e.currentTarget.style.color = "var(--text-faint)")}
          >
            {visible ? "HIDE" : "SHOW"}
          </button>
        )}
      </div>
    </div>
  );
}

function WarpNumberInput({ label, value, onChange, placeholder }: {
  label: string; value: number; onChange: (v: number) => void; placeholder?: string;
}) {
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <input type="number" value={value} onChange={(e) => onChange(Number(e.target.value))} placeholder={placeholder} className="warp-input" />
    </div>
  );
}

// ── Test result ───────────────────────────────────────────────────
function TestResultBadge({ result }: { result: TestResult | null }) {
  if (!result) return null;
  const isSuccess = result.type === "success";
  return (
    <div style={{
      marginTop: 14, padding: "11px 16px", borderRadius: 10, fontSize: 12,
      color: isSuccess ? "var(--success)" : "var(--error)",
      background: isSuccess ? "var(--success-bg)" : "var(--error-bg)",
      border: `1px solid ${isSuccess ? "rgba(52,211,153,0.15)" : "rgba(248,113,113,0.15)"}`,
      lineHeight: 1.55, animation: "fadeIn 0.18s ease-out",
      display: "flex", alignItems: "flex-start", gap: 8,
    }}>
      <span style={{ flexShrink: 0, marginTop: 1 }}>
        {isSuccess ? (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        ) : (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
          </svg>
        )}
      </span>
      <span>{result.message}</span>
    </div>
  );
}

// ── SVG Icons ─────────────────────────────────────────────────────
const IconLLM = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="3" width="20" height="14" rx="2" /><line x1="8" y1="21" x2="16" y2="21" /><line x1="12" y1="17" x2="12" y2="21" />
  </svg>
);

const IconReranker = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 7h10" /><path d="M3 12h18" /><path d="M3 17h14" /><circle cx="18" cy="7" r="2" /><circle cx="8" cy="17" r="2" />
  </svg>
);

const IconReleaseNotes = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <polyline points="10 9 9 9 8 9" />
  </svg>
);

const IconData = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="12" cy="5" rx="9" ry="3" />
    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
  </svg>
);

// ── Toggle ────────────────────────────────────────────────────────
function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label style={{ position: "relative", display: "inline-flex", cursor: "pointer", alignItems: "center", gap: 10 }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} style={{ position: "absolute", opacity: 0, width: 0, height: 0 }} />
      <div className="warp-toggle-track" style={{
        background: checked ? "var(--accent)" : "var(--surface-alt)",
        border: `1px solid ${checked ? "var(--accent)" : "var(--border)"}`,
        boxShadow: checked ? "0 0 10px var(--accent-subtle)" : "none",
      }}>
        <div className="warp-toggle-thumb" style={{
          left: checked ? 18 : 2,
          background: checked ? "#fff" : "var(--text-faint)",
        }} />
      </div>
    </label>
  );
}

// ── Service section card ───────────────────────────────────────────
function ServiceSection({ title, icon, config, onChange, testResult, onTest, testLabel, testDisabled, children }: {
  title: string; icon: React.ReactNode; config: ServiceConfig;
  onChange: (cfg: ServiceConfig) => void; testResult: TestResult | null;
  onTest: () => void; testLabel: string; testDisabled?: boolean;
  children?: React.ReactNode;
}) {
  const [testing, setTesting] = useState(false);

  const handleTest = async () => {
    setTesting(true);
    await onTest();
    setTesting(false);
  };

  return (
    <div className="warp-card" style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 22 }}>
        <div style={{
          width: 32, height: 32, borderRadius: 9,
          background: "var(--accent-subtle)",
          border: "1px solid var(--border-glow)",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "var(--accent-text)",
        }}>
          {icon}
        </div>
        <h2 style={{ fontSize: 15, fontWeight: 500, color: "var(--text-primary)", margin: 0, letterSpacing: "-0.1px" }}>
          {title}
        </h2>
      </div>
      {children}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <WarpInput label="Base URL" value={config.base_url} onChange={(v) => onChange({ ...config, base_url: v })} placeholder="https://api.openai.com/v1" />
        <WarpInput label="API Key" value={config.api_key} onChange={(v) => onChange({ ...config, api_key: v })} placeholder="sk-..." type="password" />
        <WarpInput label="Model" value={config.model} onChange={(v) => onChange({ ...config, model: v })} placeholder="gpt-4o" />
      </div>
      <div style={{ marginTop: 20 }}>
        <button
          type="button"
          disabled={testDisabled || testing}
          onClick={handleTest}
          className="btn-ghost"
          style={{ fontSize: 12, padding: "7px 18px", opacity: testDisabled ? 0.4 : 1, cursor: testDisabled ? "not-allowed" : "pointer" }}
        >
          {testing ? (
            <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <span className="spin" style={{ display: "inline-block", width: 11, height: 11, border: "1.5px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%" }} />
              Testing…
            </span>
          ) : testLabel}
        </button>
        <TestResultBadge result={testResult} />
      </div>
    </div>
  );
}

// ── Settings page ─────────────────────────────────────────────────
type Tab = "llm" | "data" | "release_notes";

const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "llm", label: "LLM", icon: <IconLLM /> },
  { id: "data", label: "Data", icon: <IconData /> },
  { id: "release_notes", label: "Release Note", icon: <IconReleaseNotes /> },
];

export default function Settings() {
  const [settings, setSettings] = useState<SettingsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [llmTest, setLlmTest] = useState<TestResult | null>(null);
  const [rerankerTest, setRerankerTest] = useState<TestResult | null>(null);
  const [original, setOriginal] = useState<SettingsData | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("llm");

  // ── Data tab state ─────────────────────────────────────────────
  const [backupStats, setBackupStats] = useState<BackupStats | null>(null);
  const [exporting, setExporting] = useState(false);
  const [importing, setImporting] = useState(false);
  const [dataMsg, setDataMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [importPreview, setImportPreview] = useState<PreviewResult | null>(null);
  const [pendingImportPath, setPendingImportPath] = useState<string | null>(null);
  const [confirmReplace, setConfirmReplace] = useState(false);

  useEffect(() => {
    apiGet<SettingsData>("/api/settings")
      .then((data) => { setSettings(data); setOriginal(data); })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const testLlm = useCallback(async () => {
    if (!settings) return;
    setLlmTest(null);
    try {
      const res = await apiPost<{ success: boolean; response?: string; error?: string }>("/api/settings/test-llm", {
        base_url: settings.llm.base_url || "https://api.openai.com/v1", api_key: settings.llm.api_key, model: settings.llm.model,
      });
      setLlmTest(res.success
        ? { type: "success", message: `LLM responded: "${res.response}"` }
        : { type: "error", message: res.error ?? "Unknown error" });
    } catch (err: unknown) {
      setLlmTest({ type: "error", message: (err as Error).message });
    }
  }, [settings]);

  const testReranker = useCallback(async () => {
    if (!settings) return;
    setRerankerTest(null);
    try {
      const res = await apiPost<{ success: boolean; results?: Array<{ index: number; score: number }>; error?: string }>(
        "/api/settings/test-reranker",
        {
          base_url: settings.reranker.base_url || "https://api.jina.ai/v1/rerank",
          api_key: settings.reranker.api_key,
          model: settings.reranker.model,
          documents: [
            "Installation guide: run the setup command and verify dependencies.",
            "Troubleshooting: if the service does not start, inspect the logs.",
          ],
          query: "How do I install the service?",
        },
      );
      setRerankerTest(
        res.success
          ? { type: "success", message: `Reranker returned ${res.results?.length ?? 0} results.` }
          : { type: "error", message: res.error ?? "Unknown error" },
      );
    } catch (err: unknown) {
      setRerankerTest({ type: "error", message: (err as Error).message });
    }
  }, [settings]);



  const save = useCallback(async () => {
    if (!settings || !original) return;
    setSaving(true); setSaveStatus(null);
    try {
      const payload: Partial<SettingsData> = {};

      const llmBaseUrl = settings.llm.base_url || "https://api.openai.com/v1";
      const llmChanged = llmBaseUrl !== original.llm.base_url || settings.llm.model !== original.llm.model || settings.llm.api_key !== original.llm.api_key;
      if (llmChanged) {
        payload.llm = { base_url: llmBaseUrl, model: settings.llm.model, api_key: settings.llm.api_key };
      }

      const rerankerBaseUrl = settings.reranker.base_url || "https://api.jina.ai/v1/rerank";
      const rerankerChanged =
        settings.reranker.enabled !== original.reranker.enabled ||
        rerankerBaseUrl !== original.reranker.base_url ||
        settings.reranker.model !== original.reranker.model ||
        settings.reranker.api_key !== original.reranker.api_key ||
        settings.reranker.top_n !== original.reranker.top_n ||
        settings.reranker.candidate_k !== original.reranker.candidate_k ||
        settings.reranker.timeout_s !== original.reranker.timeout_s;
      if (rerankerChanged) {
        payload.reranker = {
          enabled: settings.reranker.enabled,
          base_url: rerankerBaseUrl,
          model: settings.reranker.model,
          api_key: settings.reranker.api_key,
          top_n: settings.reranker.top_n,
          candidate_k: settings.reranker.candidate_k,
          timeout_s: settings.reranker.timeout_s,
        };
      }

      const updated = await apiPut<SettingsData>("/api/settings", payload);
      setSettings(updated); setOriginal(updated);
      setSaveStatus("saved");
    } catch (err: unknown) {
      setSaveStatus(`error:${(err as Error).message}`);
    } finally { setSaving(false); }
  }, [settings, original]);

  // ── Dirty check ───────────────────────────────────────────────
  const isDirty = (() => {
    if (!settings || !original) return false;
    const llmBaseUrl = settings.llm.base_url || "https://api.openai.com/v1";
    if (llmBaseUrl !== original.llm.base_url) return true;
    if (settings.llm.model !== original.llm.model) return true;
    if (settings.llm.api_key !== original.llm.api_key) return true;
    const rerankerBaseUrl = settings.reranker.base_url || "https://api.jina.ai/v1/rerank";
    if (settings.reranker.enabled !== original.reranker.enabled) return true;
    if (rerankerBaseUrl !== original.reranker.base_url) return true;
    if (settings.reranker.model !== original.reranker.model) return true;
    if (settings.reranker.api_key !== original.reranker.api_key) return true;
    if (settings.reranker.top_n !== original.reranker.top_n) return true;
    if (settings.reranker.candidate_k !== original.reranker.candidate_k) return true;
    if (settings.reranker.timeout_s !== original.reranker.timeout_s) return true;

    return false;
  })();

  if (loading) return (
    <div style={{ display: "flex", flex: 1, alignItems: "center", justifyContent: "center" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
        <div style={{ width: 18, height: 18, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%" }} className="spin" />
        <p style={{ color: "var(--text-faint)", fontSize: 13 }}>Loading settings…</p>
      </div>
    </div>
  );

  if (error) return (
    <div style={{ display: "flex", flex: 1, alignItems: "center", justifyContent: "center" }}>
      <p style={{ color: "var(--error)", fontSize: 14 }}>Error: {error}</p>
    </div>
  );

  if (!settings) return null;

  const isSaveError = saveStatus?.startsWith("error:");
  const saveMsg = isSaveError ? saveStatus!.slice(6) : saveStatus === "saved" ? "Settings saved." : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Page header */}
      <div style={{ padding: "28px 40px 0", borderBottom: "1px solid var(--border)", flexShrink: 0 }} className="fade-in">
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 22 }}>
          <div>
            <h1 className="heading-display" style={{ margin: "0 0 6px" }}>Settings</h1>
            <p style={{ fontSize: 13, color: "var(--text-muted)", margin: 0 }}>
              Configure LLM provider and general settings.
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 4 }}>
            {saveMsg && (
              <span style={{
                fontSize: 12, color: isSaveError ? "var(--error)" : "var(--success)",
                animation: "fadeIn 0.2s ease-out",
                display: "inline-flex", alignItems: "center", gap: 5,
              }}>
                {!isSaveError && (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
                {saveMsg}
              </span>
            )}
            <button
              type="button"
              disabled={saving || !isDirty}
              onClick={save}
              className="btn-accent"
              style={{ opacity: (!isDirty && !saving) ? 0.4 : 1, padding: "8px 20px" }}
            >
              {saving ? (
                <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <span className="spin" style={{ display: "inline-block", width: 12, height: 12, border: "1.5px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", borderRadius: "50%" }} />
                  Saving…
                </span>
              ) : "Save Settings"}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 0 }}>
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: "flex", alignItems: "center", gap: 7,
                  padding: "10px 18px", fontSize: 13, fontWeight: isActive ? 500 : 400,
                  color: isActive ? "var(--text-primary)" : "var(--text-muted)",
                  background: "none", border: "none", cursor: "pointer",
                  borderBottom: isActive ? "2px solid var(--accent)" : "2px solid transparent",
                  marginBottom: -1,
                  transition: "all 0.15s",
                  fontFamily: "inherit",
                }}
                onMouseEnter={e => { if (!isActive) e.currentTarget.style.color = "var(--text-secondary)"; }}
                onMouseLeave={e => { if (!isActive) e.currentTarget.style.color = "var(--text-muted)"; }}
              >
                <span style={{ color: isActive ? "var(--accent-text)" : "var(--text-faint)" }}>{tab.icon}</span>
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "28px 40px" }}>
        <div style={{ maxWidth: 600 }} className="fade-in" key={activeTab}>

          {activeTab === "llm" && (
            <>
              <ServiceSection
                title="Language Model (OpenAI-compatible API)"
                icon={<IconLLM />}
                config={settings.llm}
                onChange={(cfg) => setSettings({ ...settings, llm: cfg })}
                testResult={llmTest}
                onTest={testLlm}
                testLabel="Test connection"
              />
              <ServiceSection
                title="Reranker (Cross-encoder API)"
                icon={<IconReranker />}
                config={settings.reranker}
                onChange={(cfg) => setSettings({ ...settings, reranker: { ...settings.reranker, ...cfg } })}
                testResult={rerankerTest}
                onTest={testReranker}
                testLabel="Test reranker"
              >
                <div style={{ marginBottom: 18, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 13, color: "var(--text-primary)", marginBottom: 3 }}>Enable reranker</div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Reorder RRF candidates before building chat context.</div>
                  </div>
                  <Toggle checked={settings.reranker.enabled} onChange={(v) => setSettings({ ...settings, reranker: { ...settings.reranker, enabled: v } })} />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 16 }}>
                  <WarpNumberInput label="Top N" value={settings.reranker.top_n} onChange={(v) => setSettings({ ...settings, reranker: { ...settings.reranker, top_n: v } })} />
                  <WarpNumberInput label="Candidate K" value={settings.reranker.candidate_k} onChange={(v) => setSettings({ ...settings, reranker: { ...settings.reranker, candidate_k: v } })} />
                  <WarpNumberInput label="Timeout (seconds)" value={settings.reranker.timeout_s} onChange={(v) => setSettings({ ...settings, reranker: { ...settings.reranker, timeout_s: v } })} />
                </div>
              </ServiceSection>
            </>
          )}

          {activeTab === "data" && <DataTab
            stats={backupStats}
            setStats={setBackupStats}
            exporting={exporting}
            setExporting={setExporting}
            importing={importing}
            setImporting={setImporting}
            dataMsg={dataMsg}
            setDataMsg={setDataMsg}
            importPreview={importPreview}
            setImportPreview={setImportPreview}
            pendingImportPath={pendingImportPath}
            setPendingImportPath={setPendingImportPath}
            confirmReplace={confirmReplace}
            setConfirmReplace={setConfirmReplace}
          />}


          {activeTab === "release_notes" && (
            <div className="warp-card">
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 22 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: 9,
                  background: "var(--accent-subtle)",
                  border: "1px solid var(--border-glow)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: "var(--accent-text)",
                }}>
                  <IconReleaseNotes />
                </div>
                <h2 style={{ fontSize: 15, fontWeight: 500, color: "var(--text-primary)", margin: 0 }}>Release Notes (v1.0)</h2>
              </div>
              <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
                <h3 style={{ fontSize: 14, color: "var(--text-primary)", marginTop: 0, marginBottom: 10 }}>Version 1.0 - Welcome to LAIDocs!</h3>
                <p style={{ marginBottom: 12, lineHeight: 1.6 }}>Welcome to the initial release of LAIDocs, your 100% local AI-powered document manager. This release introduces the foundational capabilities:</p>

                <h4 style={{ fontSize: 13, color: "var(--text-primary)", marginTop: 16, marginBottom: 8 }}>✨ Core Features</h4>
                <ul style={{ paddingLeft: 20, margin: 0, display: "flex", flexDirection: "column", gap: 8 }}>
                  <li><strong>Convert Documents to Markdown</strong>: Seamlessly upload complex files like PDF, DOCX, and PPTX. LAIDocs automatically extracts text, layouts, and tables into a clean, editable Markdown format.</li>
                  <li><strong>Web Crawling</strong>: Paste any URL to intelligently extract webpage content into readable Markdown, stripping away ads and unnecessary clutter.</li>
                  <li><strong>Chat with Documents</strong>: Engage with a smart, DeepAgents-powered AI assistant. It answers questions <em>strictly</em> based on the document's context, remembers conversation history, and manages separate chat sessions.</li>
                </ul>

                <h4 style={{ fontSize: 13, color: "var(--text-primary)", marginTop: 20, marginBottom: 8 }}>🛠️ Tech Stack</h4>
                <ul style={{ paddingLeft: 20, margin: 0, display: "flex", flexDirection: "column", gap: 8 }}>
                  <li><strong>Frontend & Shell</strong>: Tauri v2 (Rust), React 19, TypeScript, Tailwind CSS, ByteMD.</li>
                  <li><strong>Backend Core</strong>: Python FastAPI, SQLite (for metadata, tree indexes, and chat history).</li>
                  <li><strong>AI & Pipelines</strong>: Docling (Conversion), Crawl4AI (Web extraction), LangChain, LangGraph, and DeepAgents (Agentic AI framework).</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
