import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info as InfoIcon, Upload, Download, ChevronLeft, ChevronRight, Save, Filter, CheckCircle2, XCircle } from "lucide-react";
import Papa from "papaparse";

/**
 * Human Eval Frontend (CSV → Annotate → Export)
 * - Upload benchmark CSV
 * - Annotate one item at a time (Yes/No/Unsure)
 * - Autosaves per annotator in localStorage + optional JSON autobackup
 * - Export annotations to CSV/JSON; import/merge CSV/JSON to resume
 * - Visualizers:
 *   • incorrect_answers: numbered list (expects 3 strings)
 *   • related_conversation_snippet: chat bubbles (user right, assistant left)
 *   • user_query: content-only view (extracts message.content)
 * - Keyboard shortcuts: ← (Prev), → (Next), E (Export), S (Save)
 */

// ===== Types =====
interface Row {
  expanded_persona?: string;
  user_query?: string;
  correct_answer?: string;
  incorrect_answers?: string | string[]; // may be JSON-encoded in CSV
  topic_query?: string;
  preference?: string;
  topic_preference?: string;
  conversation_scenario?: string;
  pref_type?: string;
  related_conversation_snippet?: string | Array<{ role: string; content: string }>; // JSON-encoded array of messages
  who?: string;
  updated?: string;
  prev_pref?: string;
  [k: string]: any;
}

interface QuestionSpec { key: keyof Annotation; label: string }
interface VerifySpec { key: keyof Annotation; label: string }

interface Annotation {
  annotator: string;
  item_index: number;
  // Human judgment questions (right column)
  q_natural?: "Yes" | "No" | "Unsure";
  q_can_infer_pref?: "Yes" | "No" | "Unsure";
  q_well_formatted?: "Yes" | "No" | "Unsure";
  q_correct_from_gt_only?: "Yes" | "No" | "Unsure";
  q_prefer_specific_over_general?: "Yes" | "No" | "Unsure";
  // Field verification (only topic fields per latest spec)
  q_topic_query_makes_sense?: "Yes" | "No" | "Unsure";
  q_topic_preference_makes_sense?: "Yes" | "No" | "Unsure";
  // Legacy tolerated on CSV import
  v_conversation_scenario?: "Yes" | "No" | "Unsure";
  v_pref_type?: "Yes" | "No" | "Unsure";
  v_who?: "Yes" | "No" | "Unsure";
}

// ===== Columns we show (left pane) =====
const PREFERRED_COLUMNS: Array<keyof Row> = [
  "expanded_persona",
  "user_query",
  "correct_answer",
  "incorrect_answers",
  "topic_query",
  "preference",
  "topic_preference",
  "conversation_scenario",
  "pref_type",
  "related_conversation_snippet",
  "who",
  "updated",
  "prev_pref",
];

const QUESTIONS: QuestionSpec[] = [
  { key: "q_natural", label: "Is the conversation natural and believable?" },
  { key: "q_can_infer_pref", label: "From the conversation alone, can you infer the ground‑truth preference/persona?" },
  { key: "q_well_formatted", label: "Are the user query, correct answer, and incorrect answers clearly and properly formatted?" },
  { key: "q_correct_from_gt_only", label: "Given the ground‑truth preference, is the correct answer supported while the incorrect options are not?" },
  { key: "q_prefer_specific_over_general", label: "If you shared this ground‑truth preference, would you prefer this answer over a generic response?" },
];

const VERIFY_FIELDS: VerifySpec[] = [
  { key: "q_topic_query_makes_sense", label: "Does the topic_query make sense?" },
  { key: "q_topic_preference_makes_sense", label: "Does the topic_preference make sense?" },
];

// ===== Stable Row Key (for merge/resume) =====
const ROW_KEY_FIELDS: Array<keyof Row> = [
  "expanded_persona",
  "user_query",
  "correct_answer",
  "preference",
  "topic_query",
  "topic_preference",
  "conversation_scenario",
  "pref_type",
  "who",
  "prev_pref",
];

// ===== Helpers (pure) =====
const HIDDEN_LEFT_HEADERS = [
  "Does the conversation sound natural and realistic?",
  "Can the ground truth user preference be inferred from the conversation?",
  "Are the user query, correct answer, and incorrect answers well formatted?",
  "Can the correct answer be inferred from the ground-truth user preference, but not from incorrect options?",
  "As a user, if you hold the same groundtruth preference, would you prefer this model response over a more general response?",
  "Does the topic_query make sense?",
  "Does the topic_preference make sense?",
];

function isHiddenLeft(col: string) {
  const norm = col.trim().toLowerCase();
  return HIDDEN_LEFT_HEADERS.some((h) => h.trim().toLowerCase() === norm);
}

function friendlyLabel(col: string) {
  if (col === "preference") return "GROUND-TRUTH PREFERENCE/PERSONA";
  return col;
}

function parseMaybeJSON<T = any>(val: any): any {
  if (Array.isArray(val)) return val;
  if (typeof val === "string") {
    const trimmed = val.trim();
    if ((trimmed.startsWith("[") && trimmed.endsWith("]")) || (trimmed.startsWith("{") && trimmed.endsWith("}"))) {
      try { return JSON.parse(trimmed); } catch {}
    }
    if (trimmed.includes("||")) return trimmed.split("||").map((s) => s.trim()).filter(Boolean);
    if (trimmed.includes(";")) return trimmed.split(";").map((s) => s.trim()).filter(Boolean);
  }
  return val;
}

function toStringArray(x: any): string[] {
  if (Array.isArray(x)) return x.map((e) => String(e));
  if (typeof x === "string" && x) return [x];
  return [];
}

function toMessageArray(x: any): Array<{ role: string; content: string }> {
  if (Array.isArray(x)) return x as any;
  if (typeof x === "string" && x) {
    const p = parseMaybeJSON(x);
    if (Array.isArray(p)) return p as any;
  }
  return [];
}

function computeRowKey(row: Row): string {
  const picked: Record<string, any> = {};
  ROW_KEY_FIELDS.forEach((f) => (picked[f as string] = (row as any)?.[f] ?? ""));
  const raw = JSON.stringify(picked);
  let h = 5381;
  for (let i = 0; i < raw.length; i++) h = (h * 33) ^ raw.charCodeAt(i);
  const hex = (h >>> 0).toString(16);
  return hex;
}

function extractContent(val: any): string {
  const v = parseMaybeJSON(val);
  if (Array.isArray(v)) {
    const contents = (v as any[])
      .map((m) => (m && typeof m === "object" && "content" in m ? String((m as any).content ?? "") : String(m ?? "")))
      .filter(Boolean);
    return contents.join("\n\n"); // normalized join
  }
  if (v && typeof v === "object" && "content" in v) return String((v as any).content ?? "");
  if (typeof v === "string") return v;
  return "";
}

function extractSingleMessageContent(val: any): string {
  // For values like: "{'role': 'user', 'content': '...'}" or "{\"role\":\"user\",\"content\":\"...\"}"
  const raw = extractContent(val);
  if (typeof raw !== "string") return String(raw ?? "");
  const mSingle = raw.match(/'content'\s*:\s*'(.*?)'/s);
  if (mSingle) return mSingle[1];
  const mDouble = raw.match(/"content"\s*:\s*"(.*?)"/s);
  if (mDouble) return mDouble[1];
  return raw;
}

// ===== Hook: useLocalStorage =====
function useLocalStorage<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {}
  }, [key, value]);
  return [value, setValue] as const;
}

// ===== App =====
export default function App() {
  const [displayCols, setDisplayCols] = useState<Array<keyof Row>>(PREFERRED_COLUMNS as any);
  const leftDisplayCols = useMemo(() => displayCols.filter((c) => !isHiddenLeft(String(c))), [displayCols]);
  const [annotator, setAnnotator] = useLocalStorage<string>("he_annotator", "");
  const [rows, setRows] = useState<Row[]>([]);
  const [workingSetSize, setWorkingSetSize] = useLocalStorage<number>("he_workingsize", 100);
  const [order, setOrder] = useLocalStorage<number[]>("he_order", []);
  const [currentIdx, setCurrentIdx] = useLocalStorage<number>("he_current", 0);
  const [annotations, setAnnotations] = useLocalStorage<Record<number, Annotation>>("he_annotations", {});
  const [annByKey, setAnnByKey] = useLocalStorage<Record<string, Annotation>>("he_ann_bykey", {});
  const [autoBackup, setAutoBackup] = useLocalStorage<boolean>("he_autobackup", true);
  const [onlyUnfinished, setOnlyUnfinished] = useLocalStorage<boolean>("he_onlyunfinished", false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const importAnnInputRef = useRef<HTMLInputElement>(null);

  // Export panel state (for sandboxed environments where downloads are blocked)
  const [exportCsv, setExportCsv] = useState<string>("");
  const [exportHref, setExportHref] = useState<string>("");
  const [exportFilename, setExportFilename] = useState<string>("annotations.csv");
  const [showExportPanel, setShowExportPanel] = useState<boolean>(false);

  const visibleOrder = useMemo(() => {
    if (!onlyUnfinished) return order;
    return order.filter((i) => !annotations[i]);
  }, [order, annotations, onlyUnfinished]);

  const progress = useMemo(() => {
    const done = Object.keys(annotations).length;
    const total = Math.min(workingSetSize, rows.length);
    return total ? Math.round((done / total) * 100) : 0;
  }, [annotations, workingSetSize, rows.length]);

  const activeGlobalIndex = visibleOrder[currentIdx] ?? 0;
  const activeRow: Row | undefined = rows[activeGlobalIndex];

  function resetWorkflow(parsedRows: Row[]) {
    const size = Math.min(workingSetSize, parsedRows.length || 0);
    const idxs = [...Array(parsedRows.length).keys()]; // preserve CSV order
    setRows(parsedRows);
    setOrder(idxs.slice(0, size));
    setCurrentIdx(0);
    setAnnotations({});
  }

  function handleFile(file: File) {
    Papa.parse<Row>(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h) => h.trim(),
      complete: (res) => {
        const raw = (res.data || []).map((r) => ({ ...(r as any) } as Row));
        const data = raw.map((row) => {
          row.incorrect_answers = parseMaybeJSON(row.incorrect_answers);
          row.related_conversation_snippet = parseMaybeJSON(row.related_conversation_snippet);
          return row;
        });
        const headers = raw.length ? Object.keys(raw[0]) : [];
        const preferred = PREFERRED_COLUMNS.filter((c) => headers.includes(c as string));
        const extras = headers.filter((h) => !(PREFERRED_COLUMNS as any).includes(h));
        setDisplayCols([...(preferred as any), ...(extras as any)]);
        resetWorkflow(data);
        prefillAnnotationsFromKeys(data);
      },
      error: (err) => alert("Failed to parse CSV: " + err.message),
    });
  }

  function triggerUpload() { fileInputRef.current?.click(); }

  function setAnnotation(partial: Partial<Annotation>) {
    if (activeRow == null) return;
    setAnnotations((prev) => ({
      ...prev,
      [activeGlobalIndex]: {
        annotator,
        item_index: activeGlobalIndex,
        ...(prev[activeGlobalIndex] || {}),
        ...partial,
      },
    }));
    const key = computeRowKey(activeRow);
    setAnnByKey((prev) => ({
      ...prev,
      [key]: {
        annotator,
        item_index: activeGlobalIndex,
        ...(prev[key] || {}),
        ...partial,
      },
    }));
  }

  function nav(delta: number) {
    setCurrentIdx((i) => {
      const next = i + delta;
      if (next < 0) return 0;
      if (next >= visibleOrder.length) return Math.max(visibleOrder.length - 1, 0);
      return next;
    });
  }

  function exportCSV() {
    // Build a CSV of the current rows + answers so it can be imported later to resume.
    if (!rows.length) {
      // Fallback: export a minimal CSV from any existing annByKey so users can still resume later.
      const minimal = Object.entries(annByKey || {}).map(([key, a]) => ({
        row_key: key,
        annotator: (a as any)?.annotator ?? annotator,
        item_index: (a as any)?.item_index ?? "",
        q_natural: (a as any)?.q_natural ?? "",
        q_can_infer_pref: (a as any)?.q_can_infer_pref ?? "",
        q_well_formatted: (a as any)?.q_well_formatted ?? "",
        q_correct_from_gt_only: (a as any)?.q_correct_from_gt_only ?? "",
        q_prefer_specific_over_general: (a as any)?.q_prefer_specific_over_general ?? "",
        q_topic_query_makes_sense: (a as any)?.q_topic_query_makes_sense ?? "",
        q_topic_preference_makes_sense: (a as any)?.q_topic_preference_makes_sense ?? "",
        v_conversation_scenario: (a as any)?.v_conversation_scenario ?? "",
        v_pref_type: (a as any)?.v_pref_type ?? "",
        v_who: (a as any)?.v_who ?? "",
      }));
      const fallbackCSV = Papa.unparse(minimal.length ? minimal : [{ row_key: "", annotator, item_index: "" }]);
      forceDownloadCSV(fallbackCSV, `annotations_${annotator || "anon"}.csv`);
      return;
    }

    const total = Math.min(workingSetSize, rows.length);
    const output = visibleOrder.length === order.length ? order.slice(0, total) : visibleOrder;

    const records = output.map((gi) => {
      const r = rows[gi] || ({} as Row);
      const a = annotations[gi] || { annotator, item_index: gi };
      const base: Record<string, any> = {};
      const key = computeRowKey(r);
      base["row_key"] = key;
      displayCols.forEach((c) => (base[c] = (r as any)[c] ?? ""));
      QUESTIONS.forEach((q) => (base[q.key as string] = (a as any)[q.key] ?? ""));
      VERIFY_FIELDS.forEach((v) => (base[v.key as string] = (a as any)[v.key] ?? ""));
      base["annotator"] = a.annotator ?? annotator;
      base["item_index"] = gi;
      return base;
    });

    if (!records.length) {
      const headerObj: Record<string, any> = { row_key: "" };
      displayCols.forEach((c) => (headerObj[c as string] = ""));
      QUESTIONS.forEach((q) => (headerObj[q.key as string] = ""));
      VERIFY_FIELDS.forEach((v) => (headerObj[v.key as string] = ""));
      headerObj["annotator"] = annotator;
      headerObj["item_index"] = "";
      records.push(headerObj);
    }

    const csv = Papa.unparse(records);
    const name = `annotations_${annotator || "anon"}.csv`;
    // Always prepare data URL + panel so users can copy/save even if downloads are blocked
    const href = "data:text/csv;charset=utf-8," + encodeURIComponent("\uFEFF" + csv);
    setExportCsv(csv);
    setExportHref(href);
    setExportFilename(name);
    setShowExportPanel(true);
    // Try a best-effort direct download; if blocked, the panel remains as fallback
    try { forceDownloadCSV(csv, name); } catch {}
  }

  function forceDownloadCSV(csv: string, filename: string) {
    // Try several strategies so at least one works across browsers, extensions, and CSPs.
    try {
      const BOM = "\uFEFF"; // Excel-friendly
      const blob = new Blob([BOM, csv], { type: "text/csv;charset=utf-8" });

      // (A) File System Access API
      const anyWin = window as any;
      if (typeof anyWin.showSaveFilePicker === "function") {
        (async () => {
          try {
            const handle = await anyWin.showSaveFilePicker({
              suggestedName: filename,
              types: [{ description: "CSV", accept: { "text/csv": [".csv"] } }],
              excludeAcceptAllOption: true,
            });
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
          } catch (err) {
            console.warn("showSaveFilePicker failed, falling back", err);
            // fall through to other strategies
            anchorDownload(blob, filename);
          }
        })();
        return;
      }

      // (B) Legacy Edge/IE
      const navAny: any = window.navigator as any;
      if (navAny && typeof navAny.msSaveOrOpenBlob === "function") {
        navAny.msSaveOrOpenBlob(blob, filename);
        return;
      }

      // (C) Anchor download
      anchorDownload(blob, filename);
    } catch (e) {
      console.error("CSV download failed", e);
      alert("Could not start the download. If this keeps happening, enable pop‑ups for this site or try another browser.");
    }
  }

  function anchorDownload(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.rel = "noopener";
    link.target = "_self";
    link.style.display = "none";
    document.body.appendChild(link);

    let clicked = false;
    try {
      link.click();
      clicked = true;
    } catch {
      try {
        link.dispatchEvent(new MouseEvent("click"));
        clicked = true;
      } catch {}
    }

    // Fallback: open in new tab if blocked
    const timer = setTimeout(() => {
      try { document.body.removeChild(link); } catch {}
      try { URL.revokeObjectURL(url); } catch {}
      if (!clicked) {
        try {
          const newTab = window.open(url, "_blank", "noopener");
          if (!newTab) throw new Error("popup blocked");
        } catch (err) {
          console.error("All download fallbacks failed", err);
          alert("Download was blocked by the browser. Please allow pop‑ups for this page and try again.");
        }
      }
    }, 800);

    setTimeout(() => {
      clearTimeout(timer);
      try { document.body.removeChild(link); } catch {}
      try { URL.revokeObjectURL(url); } catch {}
    }, 1500);
  }

  function exportJSONBackup() {
    const payload = { annotator, ts: Date.now(), annByKey };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `annotations_${annotator || "anon"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function prefillAnnotationsFromKeys(data: Row[]) {
    setAnnotations((prev) => {
      const merged: Record<number, Annotation> = { ...prev };
      data.forEach((row, idx) => {
        const key = computeRowKey(row);
        const a = annByKey[key];
        if (a) merged[idx] = { ...a, annotator: a.annotator || "", item_index: idx };
      });
      return merged;
    });
  }

  function importAnnotations(file: File) {
    const name = file.name.toLowerCase();
    if (name.endsWith(".json")) {
      const fr = new FileReader();
      fr.onload = () => {
        try {
          const data = JSON.parse(String(fr.result || "{}"));
          const merged = { ...(annByKey || {}), ...(data.annByKey || {}) } as Record<string, Annotation>;
          setAnnByKey(merged);
          prefillAnnotationsFromKeys(rows);
          alert("Merged annotations from JSON backup.");
        } catch (e: any) {
          alert("Failed to parse JSON annotations: " + e?.message);
        }
      };
      fr.readAsText(file);
    } else {
      Papa.parse<any>(file, {
        header: true,
        skipEmptyLines: true,
        complete: (res) => {
          const incoming = res.data as Array<Record<string, any>>;
          const merged: Record<string, Annotation> = { ...(annByKey || {}) };
          incoming.forEach((row) => {
            const key = row["row_key"] || computeRowKey(row as any);
            const a: Annotation = {
              annotator: String(row["annotator"] || ""),
              item_index: Number(row["item_index"] ?? 0),
              q_natural: row["q_natural"] || undefined,
              q_can_infer_pref: row["q_can_infer_pref"] || undefined,
              q_well_formatted: row["q_well_formatted"] || undefined,
              q_correct_from_gt_only: row["q_correct_from_gt_only"] || undefined,
              q_prefer_specific_over_general: row["q_prefer_specific_over_general"] || undefined,
              q_topic_query_makes_sense: row["q_topic_query_makes_sense"] || undefined,
              q_topic_preference_makes_sense: row["q_topic_preference_makes_sense"] || undefined,
              v_conversation_scenario: row["v_conversation_scenario"] || undefined,
              v_pref_type: row["v_pref_type"] || undefined,
              v_who: row["v_who"] || undefined,
            };
            merged[key] = { ...(merged[key] || {}), ...a };
          });
          setAnnByKey(merged);
          prefillAnnotationsFromKeys(rows);
          alert("Merged annotations from CSV.");
        },
        error: (err) => alert("Failed to import CSV: " + err.message),
      });
    }
  }

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "ArrowRight") nav(1);
      if (e.key === "ArrowLeft") nav(-1);
      if (e.key.toLowerCase() === "e") exportCSV();
      if (e.key.toLowerCase() === "s") setAnnotations((p) => ({ ...p }));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // periodic auto-backup of annByKey to localStorage (every 2 minutes)
  useEffect(() => {
    if (!autoBackup) return;
    const id = setInterval(() => {
      try {
        localStorage.setItem(
          "he_autobackup_payload",
          JSON.stringify({ annotator, ts: Date.now(), annByKey })
        );
      } catch {}
    }, 120000);
    return () => clearInterval(id);
  }, [autoBackup, annotator, annByKey]);

  // ---- Minimal self-tests (non-UI, console only) ----
  useEffect(() => {
    try {
      const s = extractSingleMessageContent("{'role': 'user', 'content': 'hello'}");
      console.assert(s === "hello", "extractSingleMessageContent (single quotes)");
      const d = extractSingleMessageContent('{"role":"user","content":"hi"}');
      console.assert(d === "hi", "extractSingleMessageContent (double quotes)");
      const k = computeRowKey({ preference: "a", user_query: "b" } as any);
      console.assert(typeof k === "string" && k.length > 0, "computeRowKey outputs hex");
      const hrefTest = "data:text/csv;charset=utf-8," + encodeURIComponent("\uFEFF" + "a,b\n1,2");
      console.assert(hrefTest.includes("%EF%BB%BF"), "BOM present in data URL");
    } catch {}
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Human Evaluation UI</h1>
          <div className="flex items-center gap-3">
            <Input
              placeholder="Annotator name or ID"
              value={annotator}
              onChange={(e) => setAnnotator(e.target.value)}
              className="w-56"
            />
            <Button variant="outline" onClick={triggerUpload}>
              <Upload className="mr-2 h-4 w-4" /> Upload CSV
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
                e.currentTarget.value = "";
              }}
            />
            <Button onClick={exportCSV}>
              <Download className="mr-2 h-4 w-4" /> Export CSV (E)
            </Button>
            <Button variant="outline" onClick={exportJSONBackup} title="Download JSON backup">
              Backup JSON
            </Button>
            <Button variant="outline" onClick={() => importAnnInputRef.current?.click()} title="Upload annotations (CSV/JSON)">
              Merge Annotations
            </Button>
            <input
              ref={importAnnInputRef}
              type="file"
              accept=".csv, text/csv, application/json"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) importAnnotations(f);
                e.currentTarget.value = "";
              }}
            />
          </div>
        </header>

        {showExportPanel && (
          <Card className="border-amber-300 ring-1 ring-amber-200">
            <CardContent className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold">Export Ready</h3>
                <div className="text-xs text-gray-500">If your browser blocks downloads, use the link or copy below.</div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <a href={exportHref} download={exportFilename} className="underline text-blue-600" target="_self" rel="noopener">Download {exportFilename}</a>
                <span className="text-xs text-gray-500">(Right‑click → Save Link As… if needed)</span>
              </div>
              <div>
                <Label className="text-xs">CSV preview (you can copy this):</Label>
                <textarea
                  className="mt-1 w-full h-40 text-xs font-mono bg-white border rounded p-2"
                  readOnly
                  value={exportCsv}
                />
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(exportCsv);
                      alert("CSV copied to clipboard.");
                    } catch {
                      // Fallback: select the textarea programmatically
                      const ta = document.querySelector('textarea');
                      if (ta) {
                        (ta as HTMLTextAreaElement).focus();
                        (ta as HTMLTextAreaElement).select();
                        document.execCommand('copy');
                        alert("CSV copied via fallback. If it didn't work, press ⌘/Ctrl+C.");
                      }
                    }
                  }}
                >Copy CSV</Button>
                <Button variant="outline" onClick={() => setShowExportPanel(false)}>Close</Button>
              </div>
            </CardContent>
          </Card>
        )}

        <Alert>
          <InfoIcon className="h-4 w-4" />
          <AlertTitle>How to use</AlertTitle>
          <AlertDescription>
            Export your final dataset from Numbers as <b>CSV</b>, upload it, then annotate items one by one. Progress is autosaved in your browser per annotator. Use the filters to focus on unfinished items. When done, click <i>Export CSV</i> and collect files from each annotator.
          </AlertDescription>
        </Alert>

        <Card>
          <CardContent className="p-4 grid grid-cols-1 gap-4 md:grid-cols-4 items-center">
            <div className="col-span-2 flex items-center gap-2">
              <Label className="w-48">Working set size</Label>
              <Input
                type="number"
                min={1}
                max={rows.length || 100000}
                value={workingSetSize}
                onChange={(e) => setWorkingSetSize(Math.max(1, Number(e.target.value || 1)))}
                className="w-28"
                disabled={!rows.length}
              />
            </div>
            <div className="col-span-1 flex items-center gap-2">
              <Filter className="h-4 w-4" />
              <Switch checked={onlyUnfinished} onCheckedChange={setOnlyUnfinished} />
              <Label>Only unfinished</Label>
            </div>
            <div className="col-span-1">
              <Label>Progress</Label>
              <Progress value={progress} className="mt-2" />
              <div className="text-xs text-gray-600 mt-1">{Object.keys(annotations).length} / {Math.min(workingSetSize, rows.length || 0)} completed</div>
              <div className="flex items-center gap-2 mt-3">
                <Switch checked={autoBackup} onCheckedChange={setAutoBackup} />
                <Label>Auto-backup (local)</Label>
              </div>
            </div>
          </CardContent>
        </Card>

        {activeRow ? (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card className="lg:col-span-1">
              <CardContent className="space-y-4 p-4">
                <h2 className="text-lg font-semibold">Item #{activeGlobalIndex + 1}</h2>
                <div className="grid grid-cols-1 gap-3">
                  {leftDisplayCols.map((col) => (
                    <div key={col as string} className="bg-white rounded-xl p-3 shadow-sm border">
                      <div className="text-xs uppercase tracking-wide text-gray-500">{friendlyLabel(col as string)}</div>
                      <div className="whitespace-pre-wrap text-sm mt-1">
                        {col === "user_query" ? (
                          <span className="font-bold">{extractSingleMessageContent((activeRow as any)?.user_query)}</span>
                        ) : col === "incorrect_answers" ? (
                          <IncorrectAnswersList items={toStringArray((activeRow as any)?.incorrect_answers)} />
                        ) : col === "related_conversation_snippet" ? (
                          <ChatTranscript messages={toMessageArray((activeRow as any)?.related_conversation_snippet)} />
                        ) : col === "preference" ? (
                          <span className="font-bold">{(activeRow as any)?.[col] ?? ""}</span>
                        ) : (
                          <>{String((activeRow as any)?.[col] ?? "")}</>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="lg:col-span-1">
              <CardContent className="space-y-5 p-4">
                <h3 className="text-base font-semibold">Human Judgments</h3>
                <div className="space-y-4">
                  {QUESTIONS.map((q) => (
                    <YesNoTristate
                      key={q.key as string}
                      label={q.label}
                      value={(annotations[activeGlobalIndex] as any)?.[q.key] ?? ""}
                      onChange={(v) => setAnnotation({ [q.key]: v } as any)}
                    />
                  ))}
                </div>

                <h3 className="text-base font-semibold pt-2">Field Verification</h3>
                <div className="pt-2">
                  <h4 className="font-medium mb-2">Verify Fields Make Sense</h4>
                  <div className="space-y-3">
                    {VERIFY_FIELDS.map((v) => (
                      <YesNoTristate
                        key={v.key as string}
                        label={v.label}
                        value={(annotations[activeGlobalIndex] as any)?.[v.key] ?? ""}
                        onChange={(val) => setAnnotation({ [v.key]: val } as any)}
                      />
                    ))}
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2">
                  <Button variant="secondary" onClick={() => nav(-1)}>
                    <ChevronLeft className="mr-2 h-4 w-4" /> Prev (←)
                  </Button>
                  <Button variant="secondary" onClick={() => nav(1)}>
                    Next (→) <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                </div>

                <div className="flex items-center justify-between text-xs text-gray-600">
                  <div className="flex items-center gap-2">
                    <Save className="h-3.5 w-3.5" /> Autosaved locally for <b>{annotator || "(anonymous)"}</b>
                  </div>
                  <div>
                    Item {currentIdx + 1} of {visibleOrder.length} (global #{activeGlobalIndex + 1})
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <Card>
            <CardContent className="p-6 text-sm text-gray-700">
              <p className="mb-2">No data loaded yet. Upload a CSV to begin. Expected columns include:</p>
              <code className="block bg-white rounded p-3 text-xs whitespace-pre-wrap">
                {displayCols.join(", ")}
              </code>
            </CardContent>
          </Card>
        )}

        <footer className="text-xs text-gray-500 text-center py-4">
          Pro tip: Use ← / → to navigate, E to export, S to save.
        </footer>
      </div>
    </div>
  );
}

// ===== Renderers =====
function ChatTranscript({ messages }: { messages: Array<{ role: string; content: string }> }) {
  const augmented = augmentMessages(messages || []);
  if (!augmented?.length) return <div className="text-gray-500 text-sm">(no conversation)</div>;
  return (
    <div className="space-y-2">
      {augmented.map((m, i) => (
        <ChatBubble key={i} role={m.role} content={m.content} />
      ))}
    </div>
  );
}

function ChatBubble({ role, content }: { role: string; content: string }) {
  const isUser = (role || "").toLowerCase() === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm shadow-sm border ${isUser ? "bg-gray-100" : "bg-white"}`}>
        <div className="text-[10px] uppercase tracking-wide text-gray-500 mb-1">{isUser ? "User" : "Assistant"}</div>
        <div className={`whitespace-pre-wrap ${isUser ? "text-right" : ""}`}>{content}</div>
      </div>
    </div>
  );
}

const EXTRA_USER_QUERY = "What are some memorable driving routes that connect historic landmarks across several neighboring states for a summer getaway?";
function augmentMessages(messages: Array<{ role: string; content: string }>): Array<{ role: string; content: string }> {
  return [...(messages || []), { role: "user", content: EXTRA_USER_QUERY }];
}

function IncorrectAnswersList({ items }: { items: string[] }) {
  if (!items?.length) return <div className="text-gray-500 text-sm">(none)</div>;
  return (
    <ol className="list-decimal pl-5 space-y-1">
      {items.map((t, i) => (
        <li key={i} className="whitespace-pre-wrap">{t}</li>
      ))}
    </ol>
  );
}

function YesNoTristate({ label, value, onChange }: { label: string; value: "Yes" | "No" | "Unsure" | ""; onChange: (v: "Yes" | "No" | "Unsure") => void }) {
  return (
    <div className="flex items-center justify-between gap-4 bg-white rounded-xl p-3 shadow-sm border">
      <div className="text-sm font-medium flex-1">{label}</div>
      <div className="flex items-center gap-2">
        <YNButton selected={value === "Yes"} onClick={() => onChange("Yes")} icon={<CheckCircle2 className="h-4 w-4" />}>Yes</YNButton>
        <YNButton selected={value === "No"} onClick={() => onChange("No")} icon={<XCircle className="h-4 w-4" />}>No</YNButton>
        <YNButton selected={value === "Unsure"} onClick={() => onChange("Unsure")} icon={<InfoIcon className="h-4 w-4" />}>Unsure</YNButton>
      </div>
    </div>
  );
}

function YNButton({ selected, onClick, children, icon }: { selected?: boolean; onClick: () => void; children: React.ReactNode; icon: React.ReactNode }) {
  return (
    <Button
      variant={selected ? "default" : "outline"}
      onClick={onClick}
      className={`min-w-[84px] justify-center ${selected ? "" : "bg-white"}`}
    >
      <span className="mr-2">{icon}</span>
      {children}
    </Button>
  );
}
