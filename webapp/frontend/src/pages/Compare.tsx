import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { api, ApiError, BenchResponse, checkBanglaInput, CompareSentResponse, CompareWordResponse, Metrics, ModelKey, ModelsResponse, SimilarityResponse } from "../api";
import WordInput from "../components/WordInput";
import { Heatmap, NeighborBars, ProvenanceBadges, Tokens } from "../components/Bits";

const ORDER: ModelKey[] = ["merged_plus", "merged", "fasttext", "news", "informal"];
const THESIS: ModelKey[] = ["merged", "fasttext", "news", "informal"];      // models with recorded thesis / CONF values
const COLOR: Record<ModelKey, string> = { merged_plus: "#ff3d6e", merged: "#c9a0aa", merged_sim: "#ff8fa3", fasttext: "#8a9aa2", news: "#2fa77a", informal: "#e8b23a" };
const fade = { initial: { opacity: 0, y: 16 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, margin: "-60px" }, transition: { duration: 0.6, ease: [0.2, 0.8, 0.2, 1] } } as const;

export default function Compare() {
  const [meta, setMeta] = useState<ModelsResponse | null>(null);
  useEffect(() => { api.models().then(setMeta).catch(() => setMeta(null)); }, []);
  const name = (k: ModelKey) => meta?.models[k].name ?? k;
  return (
    <div className="page">
      <div className="page-bg" />
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
        <div className="eyebrow">Compare · same question, every model</div>
        <h1 className="h-display">BanglaMeta against its rivals</h1>
        <p className="lede">Every list, score and chart below is computed on request from the served models:
          <b style={{ color: COLOR.merged_plus }}>BanglaMeta+</b> (one vector per word built from the frozen merged rows: the thesis
          rows plus their similarity-tuned view), the thesis model <b style={{ color: COLOR.merged }}>BanglaMeta</b> itself, and
          <b style={{ color: COLOR.fasttext }}> fastText</b>, <b style={{ color: COLOR.news }}>News-W2V</b> and <b style={{ color: COLOR.informal }}>Informal-W2V</b>.
          A word a model does not contain is shown as missing, never filled in.</p>
        <div className="legend" style={{ marginTop: 14 }}>
          {ORDER.map((k) => <span key={k} style={{ ["--c" as string]: COLOR[k] }}>{name(k)} · {meta ? `${meta.models[k].rows.toLocaleString()} words × ${meta.models[k].dim}` : "…"}</span>)}
        </div>
      </motion.div>

      <WordCompare name={name} />
      <PairCompare name={name} />
      <SentenceCompare name={name} />
      <Quality name={name} />
    </div>
  );
}

type NameFn = (k: ModelKey) => string;

function WordCompare({ name }: { name: NameFn }) {
  const [word, setWord] = useState("ভালো");
  const [k, setK] = useState(10);
  const [res, setRes] = useState<CompareWordResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true; setErr(null);
    api.compareWord(word, k).then((r) => live && setRes(r)).catch((e) => live && setErr(e instanceof ApiError ? e.message : String(e)));
    return () => { live = false; };
  }, [word, k]);
  return (
    <motion.section className="section" {...fade}>
      <div className="eyebrow">01 · nearest neighbours</div>
      <h2 className="h-display">One word, five neighbourhoods</h2>
      <p>The ten rows with the highest cosine to the word in each model. The overlap figure is the share of neighbours a model has in common with BanglaMeta (Jaccard).</p>
      <div className="row" style={{ marginBottom: 16 }}>
        <div style={{ width: 360 }}><WordInput onPick={(w) => setWord(w)} placeholder="বাংলা / Banglish word" /></div>
        <label className="note" style={{ display: "flex", gap: 6, alignItems: "center", paddingBottom: 12 }}>
          k <input type="range" min={5} max={30} value={k} onChange={(e) => setK(+e.target.value)} /> <span className="mono">{k}</span>
        </label>
      </div>
      {err && <div className="error">{err}</div>}
      {res && (
        <>
          <div className="note" style={{ marginBottom: 10 }}>Query: <b style={{ fontSize: 18, color: "var(--text)" }}>{res.query.word ?? res.query.input}</b>
            {res.query.kind === "banglish" && <> (typed <span className="mono">{res.query.input}</span>)</>}</div>
          <div className="grid5">
            {ORDER.map((m, i) => {
              const c = res.models[m];
              const ov = res.overlap[`merged|${m}`];
              return (
                <motion.div key={m + res.query.input + k} className={`panel mcol ${m.startsWith("merged") ? "primary" : ""}`} style={{ ["--c" as string]: COLOR[m] }}
                  initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}>
                  <div className="mname">{name(m)} <small>{c.in_model ? `row ${c.row?.toLocaleString()}` : "missing"}</small></div>
                  {m === "merged" && c.provenance && <div style={{ margin: "8px 0" }}><ProvenanceBadges p={c.provenance} /></div>}
                  {m === "merged_plus" && <div className="note" style={{ margin: "4px 0 8px" }}>one model for classification and similarity · overlap with thesis rows: <span className="mono">{ov == null ? "–" : ov.toFixed(2)}</span></div>}
                  {!m.startsWith("merged") && <div className="note" style={{ margin: "4px 0 8px" }}>overlap with BanglaMeta: <span className="mono">{ov === null || ov === undefined ? "–" : ov.toFixed(2)}</span></div>}
                  {c.in_model ? <NeighborBars items={c.neighbors} color={COLOR[m]} onPick={setWord} /> :
                    <div className="oov-card">“{res.query.word ?? res.query.input}” is not in {name(m)}’s vocabulary.</div>}
                </motion.div>
              );
            })}
          </div>
        </>
      )}
    </motion.section>
  );
}

function PairCompare({ name }: { name: NameFn }) {
  const [txt, setTxt] = useState("ঢাকা, চট্টগ্রাম\nভালো, খারাপ\nবই, পড়া\nkhub, onek\nরাজা, রানী");
  const [res, setRes] = useState<SimilarityResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const pairs = useMemo(() => txt.split("\n").map((l) => l.split(/[,،;|]/).map((x) => x.trim())).filter((p) => p.length === 2 && p[0] && p[1]) as [string, string][], [txt]);
  const run = () => {
    setErr(null);
    for (const p of pairs) for (const w of p) { const e = checkBanglaInput(w); if (e) { setErr(e); return; } }
    api.similarity(pairs.slice(0, 20)).then(setRes).catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
  };
  useEffect(run, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <motion.section className="section" {...fade}>
      <div className="eyebrow">02 · word pairs</div>
      <h2 className="h-display">How close are two words?</h2>
      <p>Cosine similarity of each pair in each model; one pair per line, separated by a comma.</p>
      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 18 }}>
        <div>
          <textarea className="textarea" value={txt} onChange={(e) => setTxt(e.target.value)} rows={7} spellCheck={false} />
          <button className="btn primary" style={{ marginTop: 8 }} onClick={run} disabled={!pairs.length}>Compare {pairs.length} pair{pairs.length === 1 ? "" : "s"}</button>
          {err && <div className="error">{err}</div>}
        </div>
        <div className="panel" style={{ padding: 6, overflowX: "auto" }}>
          {res && (
            <table className="table">
              <thead><tr><th>pair</th>{ORDER.map((m) => <th key={m} className="n" style={{ color: COLOR[m] }}>{name(m)}</th>)}</tr></thead>
              <tbody>
                {res.results.map((r, i) => {
                  const vals = ORDER.map((m) => r.cos[m]);
                  return (
                    <motion.tr key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.05 }}>
                      <td style={{ fontSize: 15.5 }}>{r.a.word ?? <s className="error">{r.a.input}</s>} · {r.b.word ?? <s className="error">{r.b.input}</s>}</td>
                      {vals.map((v, j) => (
                        <td key={j} className="n">
                          {v === null ? <span className="faint" title="a word is missing from this model">missing</span> : (
                            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                              <span style={{ display: "inline-block", width: 54, height: 6, borderRadius: 3, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                                <motion.span style={{ display: "block", height: "100%", background: COLOR[ORDER[j]] }} initial={{ width: 0 }} animate={{ width: `${Math.max(0, v) * 100}%` }} transition={{ duration: 0.6 }} />
                              </span>{v.toFixed(3)}
                            </span>
                          )}
                        </td>
                      ))}
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </motion.section>
  );
}

function SentenceCompare({ name }: { name: NameFn }) {
  const [txt, setTxt] = useState("ami tomake khub bhalobashi\nআমি তোমাকে অনেক ভালোবাসি\nখেলাটা আজ দারুণ জমেছিল\nবইটা পড়ে মন খারাপ হয়ে গেল");
  const [res, setRes] = useState<CompareSentResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const lines = txt.split("\n").map((s) => s.trim()).filter(Boolean);
  const run = () => {
    setErr(null);
    if (lines.length < 2 || lines.length > 10) { setErr("Enter 2 to 10 sentences, one per line."); return; }
    for (const l of lines) { const e = checkBanglaInput(l); if (e) { setErr(e); return; } }
    api.compareSentences(lines).then(setRes).catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
  };
  useEffect(run, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <motion.section className="section" {...fade}>
      <div className="eyebrow">03 · sentences</div>
      <h2 className="h-display">Sentence similarity, model by model</h2>
      <p>Each sentence vector is the mean of its words’ rows (the thesis sentence representation). Coverage is the share of words the model knows; missing words are left out.</p>
      <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 18, alignItems: "start" }}>
        <div>
          <textarea className="textarea" rows={6} value={txt} onChange={(e) => setTxt(e.target.value)} spellCheck={false} />
          <button className="btn primary" style={{ marginTop: 8 }} onClick={run}>Compare sentences</button>
          {err && <div className="error">{err}</div>}
          {res && <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 8 }}>
            {res.sentences.map((s, i) => <div key={i}><span className="mono faint" style={{ fontSize: 11 }}>S{i + 1} </span><Tokens tokens={s.tokens} /></div>)}
          </div>}
        </div>
        {res && (
          <div className="grid4" style={{ gridTemplateColumns: "repeat(2, minmax(0,1fr))" }}>
            {ORDER.map((m) => (
              <div key={m} className={`panel mcol ${m === "merged" ? "primary" : ""}`} style={{ ["--c" as string]: COLOR[m] }}>
                <div className="mname" style={{ marginBottom: 10 }}>{name(m)} <small>coverage {res.models[m].coverage.map((c) => `${Math.round(c * 100)}%`).join(" · ")}</small></div>
                <Heatmap m={res.models[m].similarity} labels={res.sentences.map((_, i) => `S${i + 1}`)} />
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.section>
  );
}

const METRICS: { key: string; title: string; sub: string; pct?: boolean }[] = [
  { key: "Overall_F1", title: "Overall macro-F1 (DEV)", sub: "mean of formal and informal; linear probe, 5 folds", pct: true },
  { key: "conf_Overall_F1", title: "Overall macro-F1 (held-out CONF)", sub: "tick = official thesis CONF result · exploratory for BanglaMeta+", pct: true },
  { key: "D1_F1", title: "Formal news (D1)", sub: "7-class news topic, macro-F1", pct: true },
  { key: "D2_F1", title: "Informal text (D2)", sub: "reviews · social · hate speech, mean macro-F1", pct: true },
  { key: "analogy_native_acc", title: "Word analogy", sub: "3CosAdd accuracy, 16,678 native questions", pct: true },
  { key: "sim_auc_common", title: "Similarity AUC", sub: "same-category word pairs ranked above others", pct: true },
  { key: "tok_cov_D2", title: "Informal coverage", sub: "share of informal DEV tokens the model knows", pct: true },
];

function Quality({ name }: { name: NameFn }) {
  const [b, setB] = useState<BenchResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = () => api.benchmarks().then(setB).catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
  useEffect(() => { load(); }, []);
  useEffect(() => {
    if (!b?.job.running) return;
    const id = setInterval(load, 1500);
    return () => clearInterval(id);
  }, [b?.job.running]);
  const start = () => api.runBenchmarks().then(load).catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));

  const measured = b?.measured?.results;
  const dev = b?.measured?.thesis ?? b?.thesis_dev;
  // reference values: DEV values recorded by the thesis, plus the official CONF Overall for the thesis models
  const thesis = dev && b ? (Object.fromEntries(Object.entries(dev).map(([k, v]) => [k, { ...v, conf_Overall_F1: b.official_conf.models[k as ModelKey]?.Overall_F1 }])) as unknown as Record<ModelKey, Metrics>) : undefined;
  return (
    <motion.section className="section" {...fade}>
      <div className="eyebrow">04 · measured quality</div>
      <h2 className="h-display">Quality, measured on this server</h2>
      <p>These scores are recomputed from the served model rows with the thesis’s own definitions (DEV half, plus the held-out CONF half for classification). The white tick on each bar is the value the thesis recorded, so you can see the server reproduces it.</p>
      <div className="row" style={{ alignItems: "center", marginBottom: 16, gap: 14 }}>
        <button className="btn primary" onClick={start} disabled={!!b?.job.running}>{b?.job.running ? "Measuring…" : measured ? "Measure again" : "Run benchmarks"}</button>
        {b?.measured && !b.job.running && <span className="note">last run {b.measured.created} · {Math.round(b.measured.seconds)} s</span>}
        {b?.job.running && <div style={{ width: 340 }}><div className="progress"><div style={{ width: `${b.job.progress * 100}%` }} /></div><div className="note" style={{ marginTop: 4 }}>{b.job.message}</div></div>}
        {b?.job.error && <span className="error">{b.job.error}</span>}
        {err && <span className="error">{err}</span>}
      </div>
      {!measured && !b?.job.running && <p className="muted">No measurement yet for these model files — press “Run benchmarks” (about two minutes).</p>}
      {measured && thesis && (
        <div className="metric-grid">
          {METRICS.map((mt, mi) => {
            const vals = ORDER.map((k) => measured[k]?.[mt.key] as number);
            const best = Math.max(...vals);
            const lo = Math.min(...vals) * 0.9, hi = best * 1.02;
            return (
              <motion.div key={mt.key} className="panel metric" initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: mi * 0.06 }}>
                <h4 className="h-display">{mt.title}</h4>
                <div className="sub">{mt.sub}</div>
                {ORDER.map((k, i) => {
                  const v = vals[i]; const t = thesis[k]?.[mt.key] as number | undefined;
                  const w = (x: number) => `${Math.max(2, ((x - lo) / (hi - lo)) * 100)}%`;
                  return (
                    <div className="hbar" key={k}>
                      <span style={{ color: k.startsWith("merged") ? COLOR[k] : "var(--muted)", fontWeight: k.startsWith("merged") ? 600 : 400 }}>{name(k)}</span>
                      <div className="track">
                        <motion.div className="fill" style={{ background: COLOR[k] }} initial={{ width: 0 }} whileInView={{ width: w(v) }} viewport={{ once: true }} transition={{ duration: 0.9, delay: 0.1 + i * 0.08, ease: [0.2, 0.8, 0.2, 1] }} />
                        {t !== undefined && t !== null && <div className="tick" style={{ left: w(t) }} title={`thesis: ${(t * 100).toFixed(2)}`} />}
                      </div>
                      <span className={`v ${v === best ? "best" : ""}`}>{(v * 100).toFixed(2)}</span>
                    </div>
                  );
                })}
                <Reproduced measured={measured} thesis={thesis} metric={mt.key} />
              </motion.div>
            );
          })}
        </div>
      )}
      {b && <Official b={b} name={name} />}
    </motion.section>
  );
}

function Reproduced({ measured, thesis, metric }: { measured: Record<ModelKey, Metrics>; thesis: Record<ModelKey, Metrics>; metric: string }) {
  const diffs = THESIS.map((k) => Math.abs((measured[k]?.[metric] as number) - (thesis[k]?.[metric] as number)));
  const max = Math.max(...diffs);
  const ok = max < 0.005;
  return <div style={{ marginTop: 10 }}><span className={`verdict ${ok ? "ok" : "warn"}`}>{ok ? "✓ matches thesis" : "≠ thesis"} · max |Δ| {(max * 100).toFixed(2)} pt</span></div>;
}

function Official({ b, name }: { b: BenchResponse; name: NameFn }) {
  const o = b.official_conf;
  const rows = THESIS.map((k) => ({ k, D1_F1: o.models[k].D1_F1, D2_F1: o.models[k].D2_F1, Overall_F1: o.models[k].Overall_F1 }));
  const best = Math.max(...rows.map((r) => r.Overall_F1));
  return (
    <div style={{ marginTop: 40 }}>
      <div className="eyebrow">held-out confirmation · CONF half, used once</div>
      <h2 className="h-display" style={{ fontSize: 30, margin: "6px 0" }}>The official result</h2>
      <p className="muted" style={{ maxWidth: 820 }}>Read unchanged from the thesis result files ({o.source}). The model was frozen before this half was opened.</p>
      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 18 }}>
        <div className="panel" style={{ padding: 6 }}>
          <table className="table">
            <thead><tr><th>model</th><th className="n">news D1</th><th className="n">informal D2</th><th className="n">Overall</th></tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.k}>
                <td style={{ color: COLOR[r.k], fontWeight: r.k === "merged" ? 600 : 400 }}>{name(r.k)}</td>
                <td className="n">{(r.D1_F1 * 100).toFixed(2)}</td><td className="n">{(r.D2_F1 * 100).toFixed(2)}</td>
                <td className={`n ${r.Overall_F1 === best ? "best" : ""}`}>{(r.Overall_F1 * 100).toFixed(2)}</td>
              </tr>))}
            </tbody>
          </table>
        </div>
        <div className="panel" style={{ padding: 6 }}>
          <table className="table">
            <thead><tr><th>pre-registered test</th><th className="n">Δ Overall [95% CI]</th><th className="n">Holm p</th><th /></tr></thead>
            <tbody>{o.tests.map((t) => (
              <tr key={t.comparison}>
                <td>{t.comparison.replace("T: CONC", "BanglaMeta")}</td>
                <td className="n">+{(t.delta * 100).toFixed(2)} [{(t.ci_lo * 100).toFixed(2)}, {(t.ci_hi * 100).toFixed(2)}]</td>
                <td className="n">{t.p_holm.toFixed(4)}</td>
                <td><span className={`verdict ${t.reject_H0 ? "ok" : "warn"}`}>{t.reject_H0 ? "confirmed" : "not confirmed"}</span></td>
              </tr>))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
