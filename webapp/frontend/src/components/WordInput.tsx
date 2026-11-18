import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { api, ApiError, checkBanglaInput, Resolved } from "../api";

interface Props {
  onPick: (word: string, from: Resolved) => void;
  placeholder?: string;
  disabled?: boolean;
  autoFocus?: boolean;
  compact?: boolean;
  value?: string;
  onValue?: (v: string) => void;
  label?: string;
}

/** One-word input. While typing, the server resolves the word against the merged model's vocabulary;
 *  Banglish shows its Bangla candidates (all real model words) and the user picks one. */
export default function WordInput({ onPick, placeholder, disabled, autoFocus, compact, value, onValue, label }: Props) {
  const [inner, setInner] = useState("");
  const text = value ?? inner;
  const setText = (v: string) => (onValue ? onValue(v) : setInner(v));
  const [res, setRes] = useState<Resolved | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ctrl = useRef<AbortController | null>(null);

  useEffect(() => {
    const t = text.trim();
    setRes(null);
    if (!t) { setErr(null); return; }
    const local = checkBanglaInput(t);
    if (local) { setErr(local); return; }
    if (/\s/.test(t)) { setErr("One word at a time."); return; }
    setErr(null);
    const id = setTimeout(async () => {
      ctrl.current?.abort();
      const c = new AbortController(); ctrl.current = c; setBusy(true);
      try {
        const r = await api.resolve(t, c.signal);
        if (r.tokens.length !== 1) setErr("One word at a time.");
        else setRes(r.tokens[0]);
      } catch (e) {
        if ((e as Error).name !== "AbortError") setErr(e instanceof ApiError ? e.message : String(e));
      } finally { if (ctrl.current === c) setBusy(false); }
    }, 180);
    return () => clearTimeout(id);
  }, [text]);

  const pick = (w: string) => {
    if (!res) return;
    onPick(w, res);
    setText(""); setRes(null);
  };

  return (
    <div className="field">
      {label && <label>{label}</label>}
      <input
        className={`input ${err ? "invalid" : ""}`}
        value={text}
        disabled={disabled}
        autoFocus={autoFocus}
        placeholder={placeholder ?? "বাংলা বা Banglish শব্দ — যেমন ভালো / bhalo"}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && res?.word) pick(res.word); }}
        spellCheck={false}
        aria-invalid={!!err}
      />
      {err && <div className="error">{err}</div>}
      <AnimatePresence>
        {res && (
          <motion.div className="suggest" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            {res.found ? (
              <>
                {!compact && <span className="faint">{res.kind === "banglish" ? "model words for" : "found"} <b className="mono">{res.input}</b>:</span>}
                {res.options.slice(0, compact ? 4 : 6).map((o) => (
                  <button key={o} onClick={() => pick(o)} title="Use this word">{o}</button>
                ))}
                {!compact && <span className="note">Enter = first</span>}
              </>
            ) : (
              <span className="error" style={{ margin: 0 }}>
                “{res.input}” is not in the merged model’s vocabulary{res.kind === "banglish" ? " (no Bangla spelling of it is a model word)" : ""}.
              </span>
            )}
          </motion.div>
        )}
      </AnimatePresence>
      {busy && !res && !err && <div className="note" style={{ marginTop: 6 }}>looking up…</div>}
    </div>
  );
}
