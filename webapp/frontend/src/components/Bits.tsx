import { motion } from "framer-motion";
import { Neighbor, Provenance, Resolved } from "../api";

export function NeighborBars({ items, color = "var(--green)", onPick, max }: {
  items: Neighbor[]; color?: string; onPick?: (w: string) => void; max?: number;
}) {
  return (
    <div>
      {items.slice(0, max ?? items.length).map((n, i) => (
        <motion.div className="nb" key={n.word + i}
          initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.025, duration: 0.25 }}>
          <span className="i">{i + 1}</span>
          <div className="w" style={{ color }} onClick={() => onPick?.(n.word)} title={`row ${n.rank.toLocaleString()} (frequency rank)`}>
            <motion.div className="bar" initial={{ width: 0 }} animate={{ width: `${Math.max(0, n.cos) * 100}%` }}
              transition={{ delay: 0.1 + i * 0.025, duration: 0.5, ease: [0.2, 0.8, 0.2, 1] }} />
            <span>{n.word}</span>
          </div>
          <span className="c">{n.cos.toFixed(3)}</span>
        </motion.div>
      ))}
    </div>
  );
}

export function ProvenanceBadges({ p }: { p: Provenance | null }) {
  if (!p) return null;
  const items: [keyof Provenance, string, string][] = [["news", "News", "#2fa77a"], ["informal", "Informal", "#e8b23a"], ["subs", "Subs2vec", "#e8344f"]];
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {items.map(([k, name, c]) => (
        <span key={k} className="mono" title={p[k] ? "this block is the source model's own row" : "the source model lacks this word; the block was predicted from another source by the ridge map (thesis Step 4)"}
          style={{ fontSize: 11, padding: "2px 8px", borderRadius: 999, border: `1px ${p[k] ? "solid" : "dashed"} ${c}`, color: p[k] ? c : "var(--faint)" }}>
          {name}: {p[k] ? "original" : "imputed"}
        </span>
      ))}
    </div>
  );
}

export function Tokens({ tokens }: { tokens: (Resolved & { in_model?: boolean })[] }) {
  return (
    <div className="tokens">
      {tokens.map((t, i) => {
        const ok = t.in_model ?? t.found;
        return (
          <span key={i} className={`tok ${ok ? "ok" : "miss"}`} title={ok ? `uses row “${t.word}”` : "not in the model: left out of the sentence vector"}>
            {t.word ?? t.input}
            {t.kind === "banglish" && t.word && <sub>{t.input}</sub>}
          </span>
        );
      })}
    </div>
  );
}

export function heatColor(v: number | null): string {
  if (v === null) return "rgba(255,255,255,0.04)";
  const t = Math.max(0, Math.min(1, (v + 0.1) / 1.1));
  const a = 0.12 + 0.78 * t;
  return `rgba(232, ${Math.round(178 - 120 * t)}, ${Math.round(58 + 10 * t)}, ${a})`;
}

export function Heatmap({ m, labels }: { m: (number | null)[][]; labels: string[] }) {
  const n = m.length;
  return (
    <div style={{ display: "grid", gridTemplateColumns: `28px repeat(${n}, minmax(0,1fr))`, gap: 2, alignItems: "center", maxWidth: 28 + n * 64 }}>
      <span />
      {labels.map((l, j) => <span key={j} className="mono faint" style={{ fontSize: 10, textAlign: "center" }}>{l}</span>)}
      {m.map((row, i) => [
        <span key={`l${i}`} className="mono faint" style={{ fontSize: 10 }}>{labels[i]}</span>,
        ...row.map((v, j) => (
          <motion.div key={`${i}-${j}`} title={v === null ? "no known word" : v.toFixed(4)}
            initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (i + j) * 0.03 }}
            style={{ aspectRatio: "1", borderRadius: 4, background: heatColor(v), display: "grid", placeItems: "center",
              fontFamily: "var(--font-mono)", fontSize: n > 6 ? 9 : 11, color: v !== null && v > 0.55 ? "#1b1404" : "var(--text)" }}>
            {v === null ? "–" : v.toFixed(2)}
          </motion.div>
        )),
      ])}
    </div>
  );
}
