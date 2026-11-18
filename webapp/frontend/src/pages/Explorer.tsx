import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ModelsResponse, View, api, AnalogyResponse, ApiError, checkBanglaInput, ExploreResponse, PairItem, PairResponse, Point, Projection, Relation } from "../api";
import Scene from "../components/Scene";
import WordInput from "../components/WordInput";
import { Heatmap, NeighborBars, ProvenanceBadges, Tokens } from "../components/Bits";

const MAX_WORDS = 12;

export default function Explorer() {
  const [words, setWords] = useState<string[]>([]);
  const [draft, setDraft] = useState("");
  const [sentences, setSentences] = useState<string[]>([]);
  const [onlyInput, setOnlyInput] = useState(false);   // true = map only what the user typed
  const [showRel, setShowRel] = useState(true);          // lines between the user's own items
  const [view, setView] = useState<View>("merged_plus");
  const [meta, setMeta] = useState<ModelsResponse | null>(null);   // model facts come from the server, never hard-coded
  useEffect(() => { api.models().then(setMeta).catch(() => setMeta(null)); }, []);
  const vm = meta?.models[view]; // BanglaMeta+ by default; "merged" = the frozen thesis rows
  const [k, setK] = useState(10);
  const [projection, setProjection] = useState<Projection>("pca");
  const [dims, setDims] = useState<2 | 3>(3);
  const [rotate, setRotate] = useState(true);
  const [data, setData] = useState<ExploreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<"detail" | "relations" | "similarity" | "analogy">("detail");
  // similarity tool: two slots filled by clicking points in the view (or by typing)
  const [slots, setSlots] = useState<[PairItem | null, PairItem | null]>([null, null]);
  const [activeSlot, setActiveSlot] = useState<0 | 1>(0);
  const [pairRes, setPairRes] = useState<PairResponse | null>(null);
  const [pairErr, setPairErr] = useState<string | null>(null);
  const ctrl = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!words.length && !sentences.length) { setData(null); return; }
    ctrl.current?.abort();
    const c = new AbortController(); ctrl.current = c;
    setLoading(true); setError(null);
    api.explore({ words, sentences, k, projection, dims, similar: !onlyInput, model: view }, c.signal)
      .then((d) => { setData(d); setSelected((s) => (s && d.points.some((p) => p.id === s) ? s : d.points.find((p) => p.kind === "query")?.id ?? null)); })
      .catch((e) => { if (e.name !== "AbortError") setError(e instanceof ApiError ? e.message : String(e)); })
      .finally(() => { if (ctrl.current === c) setLoading(false); });
  }, [words, sentences, k, projection, dims, onlyInput, view]);

  const addWord = useCallback((w: string) => {
    setWords((ws) => (ws.includes(w) ? ws : [...ws, w].slice(-MAX_WORDS)));
    setSelected(`w:${w}`); setTab("detail");
  }, []);

  const draftLines = useMemo(() => draft.split("\n").map((s) => s.trim()).filter(Boolean), [draft]);
  const draftError = useMemo(() => {
    if (draftLines.length > 30) return "At most 30 sentences.";
    for (const l of draftLines) { const e = checkBanglaInput(l); if (e) return e; if (l.length > 600) return "A sentence is longer than 600 characters."; }
    return null;
  }, [draftLines]);

  const itemOf = useCallback((p: Point): PairItem => (p.kind === "sentence"
    ? { kind: "sentence", text: data!.sentences[p.sentence!].text, id: p.id, label: p.label }
    : { kind: "word", text: p.label, id: p.id, label: p.label }), [data]);

  const onPoint = (p: Point) => {
    setSelected(p.id);
    if (tab === "similarity") {
      const it = itemOf(p);
      setSlots((s) => (activeSlot === 0 ? [it, s[1]] : [s[0], it]));
      setActiveSlot((a) => (a === 0 ? 1 : 0));
    } else if (tab !== "relations") setTab("detail");
  };

  useEffect(() => {
    setPairRes(null); setPairErr(null);
    const [a, b] = slots;
    if (!a || !b) return;
    let live = true;
    api.pair({ kind: a.kind, text: a.text }, { kind: b.kind, text: b.text }, view)
      .then((r) => { if (live) setPairRes(r); })
      .catch((e) => { if (live) setPairErr(e instanceof ApiError ? e.message : String(e)); });
    return () => { live = false; };
  }, [slots, view]);

  const axisNames: [string, string, string] = projection === "tsne" ? ["t-SNE 1", "t-SNE 2", "t-SNE 3"]
    : (["PC1", "PC2", "PC3"].map((n, i) => (projection === "pca" && data?.projection.explained ? `${n} · ${(data.projection.explained[i] * 100).toFixed(1)}%` : n)) as [string, string, string]);
  const pairIds: [string | null, string | null] = [slots[0]?.id ?? null, slots[1]?.id ?? null];

  // the map follows the sentence box: typing or erasing updates it after a short pause
  useEffect(() => {
    if (draftError) return;
    const id = setTimeout(() => setSentences((prev) => (prev.join("\n") === draftLines.join("\n") ? prev : draftLines)), 450);
    return () => clearTimeout(id);
  }, [draftLines, draftError]);

  const sel: Point | undefined = data?.points.find((p) => p.id === selected);
  const selWord = sel && sel.kind !== "sentence" ? data?.words.find((w) => w.word === sel.label) : undefined;
  const selSentence = sel?.kind === "sentence" ? data?.sentences[sel.sentence!] : undefined;
  const oov = data?.words.filter((w) => !w.in_model) ?? [];

  return (
    <div className="explorer">
      {loading && <motion.div className="loading-bar" initial={{ width: "0%" }} animate={{ width: "85%" }} transition={{ duration: 2.5, ease: "easeOut" }} />}
      <Scene points={data?.points ?? []} edges={data?.edges ?? []} selected={selected} autoRotate={rotate} onSelect={onPoint}
        axisNames={axisNames} dims={dims} pair={tab === "similarity" ? pairIds : [null, null]} pairCos={pairRes?.cos ?? null}
        relations={showRel ? data?.relations ?? [] : []} />

      <AnimatePresence>
        {!data?.points.length && !loading && (
          <motion.div className="hero-title" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <div className="eyebrow">{vm ? `${vm.name} · ${vm.rows.toLocaleString()} words · ${vm.dim.toLocaleString()} dimensions` : "BanglaMeta+"}</div>
            <h1 className="h-display">শব্দের আকাশ</h1>
            <p>Add a Bangla or Banglish word, or a few sentences, to see where the merged model places them.</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ------------------------------------------------ inputs */}
      <motion.div className="left" initial={{ opacity: 0, x: -24 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6, ease: [0.2, 0.8, 0.2, 1] }}>
        <div className="panel card">
          <div className="eyebrow">01 · words</div>
          <h3 className="h-display">Word field</h3>
          <WordInput onPick={(w) => addWord(w)} />
          <div className="chips" style={{ marginTop: 10 }}>
            <AnimatePresence>
              {words.map((w) => {
                const r = data?.words.find((x) => x.word === w);
                return (
                  <motion.span layout key={w} className={`chip ${r && !r.in_model ? "oov" : ""}`} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.6, opacity: 0 }}
                    onClick={() => setSelected(`w:${w}`)} style={{ cursor: "pointer" }}>
                    {w}
                    <button aria-label={`remove ${w}`} onClick={(e) => { e.stopPropagation(); setWords((ws) => ws.filter((x) => x !== w)); }}>×</button>
                  </motion.span>
                );
              })}
            </AnimatePresence>
          </div>
          {words.length >= MAX_WORDS && <div className="note" style={{ marginTop: 6 }}>Up to {MAX_WORDS} words; adding more replaces the oldest.</div>}
        </div>

        <div className="panel card" style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div className="eyebrow">02 · sentence corpus</div>
          <h3 className="h-display">Sentences</h3>
          <textarea className={`textarea ${draftError ? "invalid" : ""}`} value={draft} onChange={(e) => setDraft(e.target.value)}
            placeholder={"এক লাইনে একটি বাক্য — Bangla or Banglish\nযেমন: ami tomake khub bhalobashi\nখেলাটা আজ দারুণ জমেছিল"} spellCheck={false} style={{ flex: 1, minHeight: 110 }} />
          {draftError && <div className="error">{draftError}</div>}
          <div className="row" style={{ marginTop: 10, justifyContent: "space-between", alignItems: "center" }}>
            <span className="note">{draftLines.length} sentence{draftLines.length === 1 ? "" : "s"} · vector = mean of word rows</span>
            <span className="note mono">{draftError ? "not mapped" : draftLines.join("\n") === sentences.join("\n") ? (loading ? "mapping…" : "on the map") : "typing…"}</span>
          </div>
          {data && data.sentences.length > 0 && (
            <div className="scroll" style={{ marginTop: 12, maxHeight: 190, display: "flex", flexDirection: "column", gap: 8 }}>
              {data.sentences.map((s, i) => (
                <div key={i} onClick={() => setSelected(`s:${i}`)} style={{ cursor: "pointer", padding: 8, borderRadius: 8, background: selected === `s:${i}` ? "rgba(255,61,90,0.08)" : "transparent" }}>
                  <div className="mono faint" style={{ fontSize: 10.5 }}>S{i + 1} · {Math.round(s.coverage * 100)}% of words in model</div>
                  <Tokens tokens={s.tokens} />
                </div>
              ))}
            </div>
          )}
        </div>
      </motion.div>

      {/* ------------------------------------------------ details */}
      <motion.div className="right" initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6, delay: 0.1, ease: [0.2, 0.8, 0.2, 1] }}>
        <div className="panel card scroll" style={{ flex: 1, minHeight: 0 }}>
          <div className="seg" style={{ marginBottom: 12 }}>
            <button className={tab === "detail" ? "on" : ""} onClick={() => setTab("detail")}>Selection</button>
            <button className={tab === "similarity" ? "on" : ""} onClick={() => setTab("similarity")}>Similarity</button>
            <button className={tab === "relations" ? "on" : ""} onClick={() => setTab("relations")}>Relations</button>
            <button className={tab === "analogy" ? "on" : ""} onClick={() => setTab("analogy")}>Analogy</button>
          </div>
          {error && <div className="error" role="alert">{error}</div>}
          {oov.length > 0 && <div className="error">Not in the model: {oov.map((w) => w.input).join(", ")}</div>}
          {tab === "relations" ? (
            <RelationsPanel rel={data?.relations ?? []} points={data?.points ?? []} selected={selected} onAdd={addWord}
              onAnalogy={() => setTab("analogy")} />
          ) : tab === "similarity" ? (
            <SimilarityPanel slots={slots} active={activeSlot} setActive={setActiveSlot} setSlots={setSlots} res={pairRes} err={pairErr}
              points={data?.points ?? []} />
          ) : tab === "analogy" ? <AnalogyPanel onAdd={addWord} view={view} /> : (
            <AnimatePresence mode="wait">
              <motion.div key={selected ?? "none"} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                {selWord ? (
                  <>
                    <div className="eyebrow">nearest words · cosine</div>
                    <h3 className="h-display" style={{ fontSize: 30, margin: "4px 0" }}>{selWord.word}</h3>
                    <div className="note" style={{ marginBottom: 8 }}>
                      row {selWord.row?.toLocaleString()} of {vm ? vm.rows.toLocaleString() : "…"} · {vm?.name}{selWord.kind === "banglish" && <> · typed as <span className="mono">{selWord.input}</span></>}
                    </div>
                    <Coords p={sel!} names={axisNames} dims={dims} />
                    <ProvenanceBadges p={selWord.provenance} />
                    <div style={{ height: 10 }} />
                    <NeighborBars items={selWord.neighbors} color="var(--gold)" onPick={addWord} />
                  </>
                ) : selSentence ? (
                  <>
                    <div className="eyebrow">sentence vector</div>
                    <p style={{ fontSize: 17, margin: "6px 0 8px" }}>{selSentence.text}</p>
                    <Coords p={sel!} names={axisNames} dims={dims} />
                    <Tokens tokens={selSentence.tokens} />
                    <div className="note" style={{ margin: "8px 0" }}>{selSentence.used} of {selSentence.tokens.length} words used · nearest words to the sentence vector:</div>
                    <NeighborBars items={selSentence.nearest} color="var(--red)" onPick={addWord} />
                  </>
                ) : sel ? (
                  <>
                    <div className="eyebrow">word</div>
                    <h3 className="h-display" style={{ fontSize: 28 }}>{sel.label}</h3>
                    <Coords p={sel} names={axisNames} dims={dims} />
                    <button className="btn small" onClick={() => addWord(sel.label)}>Show its neighbours</button>
                  </>
                ) : <p className="muted">Click a point in the sky to inspect it.</p>}
              </motion.div>
            </AnimatePresence>
          )}
          {tab === "detail" && data && data.sentence_similarity.length > 1 && (
            <div style={{ marginTop: 18 }}>
              <div className="eyebrow" style={{ marginBottom: 8 }}>sentence × sentence cosine</div>
              <Heatmap m={data.sentence_similarity} labels={data.sentences.map((_, i) => `S${i + 1}`)} />
            </div>
          )}
        </div>
      </motion.div>

      {/* ------------------------------------------------ view controls */}
      <motion.div className="panel toolbar" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
        <div className="seg" role="group" aria-label="projection">
          {([["pca", "PCA · global"], ["pca-local", "PCA · local"], ["tsne", "t-SNE"]] as [Projection, string][]).map(([v, l]) => (
            <button key={v} className={projection === v ? "on" : ""} onClick={() => setProjection(v)}>{l}</button>
          ))}
        </div>
        <div className="seg">{([3, 2] as const).map((d) => <button key={d} className={dims === d ? "on" : ""} onClick={() => setDims(d)}>{d}D</button>)}</div>
        <label className="note" style={{ display: "flex", alignItems: "center", gap: 6 }}>
          k <input type="range" min={3} max={30} value={k} onChange={(e) => setK(+e.target.value)} /> <span className="mono">{k}</span>
        </label>
        <div className="seg">
          <button className={rotate ? "on" : ""} onClick={() => setRotate(!rotate)}>Rotate</button>
          <button className={showRel ? "on" : ""} onClick={() => setShowRel(!showRel)} title="lines between your own words and sentences, labelled with cosine">Relations</button>
        </div>
        <div className="seg" title="BanglaMeta+: one model, best on classification, analogy and similarity. Thesis rows: the frozen merged model as confirmed in the thesis.">
          <button className={view === "merged_plus" ? "on" : ""} onClick={() => setView("merged_plus")} aria-pressed={view === "merged_plus"}>BanglaMeta+</button>
          <button className={view === "merged" ? "on" : ""} onClick={() => setView("merged")} aria-pressed={view === "merged"}>Thesis rows</button>
        </div>
        <div className="seg" title="ON: only your words and sentences (and the words inside them). OFF: also the similar words the model finds.">
          <button className={onlyInput ? "on" : ""} onClick={() => setOnlyInput(true)} aria-pressed={onlyInput}>Only my input</button>
          <button className={!onlyInput ? "on" : ""} onClick={() => setOnlyInput(false)} aria-pressed={!onlyInput}>+ Similar words</button>
        </div>
        {data?.projection.explained && <span className="mono faint" style={{ fontSize: 10.5 }}>PC1–3 explain {(data.projection.explained.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%</span>}
      </motion.div>
    </div>
  );
}

function AnalogyPanel({ onAdd, view }: { onAdd: (w: string) => void; view: View }) {
  const [v, setV] = useState<[string, string, string]>(["রাজা", "রানী", "পুরুষ"]);
  const [res, setRes] = useState<AnalogyResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setErr(null); setBusy(true);
    try { setRes(await api.analogy(v[0].trim(), v[1].trim(), v[2].trim(), 10, view)); }
    catch (e) { setErr(e instanceof ApiError ? e.message : String(e)); setRes(null); }
    finally { setBusy(false); }
  };
  const bad = v.map((x) => (x.trim() ? checkBanglaInput(x) : "empty")).find(Boolean);
  return (
    <div>
      <div className="eyebrow">vector arithmetic · b − a + c</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 6, alignItems: "center", margin: "10px 0" }}>
        <input className="input" value={v[0]} onChange={(e) => setV([e.target.value, v[1], v[2]])} aria-label="a" />
        <span className="muted">:</span>
        <input className="input" value={v[1]} onChange={(e) => setV([v[0], e.target.value, v[2]])} aria-label="b" />
        <input className="input" value={v[2]} onChange={(e) => setV([v[0], v[1], e.target.value])} aria-label="c" />
        <span className="muted">:</span>
        <div className="input" style={{ color: "var(--gold)", minHeight: 44 }}>{res?.ok ? res.results[0]?.word : "?"}</div>
      </div>
      <button className="btn primary" disabled={busy || !!bad} onClick={run}>{busy ? "Computing…" : "Solve"}</button>
      {err && <div className="error">{err}</div>}
      {res && !res.ok && <div className="error">Not in the model: {res.missing.join(", ")}</div>}
      {res?.ok && (
        <div style={{ marginTop: 12 }}>
          <div className="note" style={{ marginBottom: 6 }}>{res.a.word} → {res.b.word} as {res.c.word} → ? · click a result to add it to the sky</div>
          <NeighborBars items={res.results} color="var(--gold)" onPick={onAdd} />
          <div className="note" style={{ marginTop: 8 }}>{res.rule}</div>
        </div>
      )}
    </div>
  );
}

function Coords({ p, names, dims }: { p: Point; names: [string, string, string]; dims: 2 | 3 }) {
  const cols = ["#ff7a7a", "#7be0a0", "#7fb2ff"];
  return (
    <div className="coords" title={`position in this view: ${names.slice(0, dims).join(", ")}`}>
      {(["X", "Y", "Z"] as const).slice(0, dims).map((a, i) => (
        <span key={a}><span style={{ color: cols[i] }}>{a}</span> {p.xyz[i] >= 0 ? "+" : ""}{p.xyz[i].toFixed(3)}</span>
      ))}
    </div>
  );
}

function sideLabel(x: PairResponse["a"]): string {
  return x.kind_of_item === "word" ? (x.word ?? x.input) : x.text;
}

function SimilarityPanel({ slots, active, setActive, setSlots, res, err, points }: {
  slots: [PairItem | null, PairItem | null]; active: 0 | 1; setActive: (a: 0 | 1) => void;
  setSlots: (s: [PairItem | null, PairItem | null]) => void; res: PairResponse | null; err: string | null; points: Point[];
}) {
  const [typed, setTyped] = useState("");
  const typedErr = typed.trim() ? checkBanglaInput(typed) : null;
  const setSlot = (i: 0 | 1, it: PairItem | null) => setSlots(i === 0 ? [it, slots[1]] : [slots[0], it]);
  const applyTyped = () => {
    const t = typed.trim();
    if (!t || typedErr) return;
    const isWord = !/\s/.test(t);
    const match = points.find((p) => p.kind !== "sentence" && p.label === t);
    setSlot(active, { kind: isWord ? "word" : "sentence", text: t, id: match?.id ?? null, label: t });
    setActive(active === 0 ? 1 : 0); setTyped("");
  };
  const pct = res?.cos == null ? 0 : (res.cos + 1) / 2;
  return (
    <div>
      <div className="eyebrow">cosine similarity · full 900-d vectors</div>
      <p className="note" style={{ margin: "6px 0 10px" }}>Click two points in the view (words or sentences), or type a word or a sentence. The score is computed by the merged model from the full vectors, not from the 3-D picture.</p>
      <div style={{ display: "grid", gap: 8 }}>
        {([0, 1] as const).map((i) => (
          <div key={i} className={`slot ${active === i ? "on" : ""}`} onClick={() => setActive(i)}>
            <div className="k">{i === 0 ? "A" : "B"}{slots[i] ? ` · ${slots[i]!.kind}` : active === i ? " · click a point" : ""}</div>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
              <span style={{ fontSize: 16 }}>{slots[i]?.label ?? <span className="faint">—</span>}</span>
              {slots[i] && <button className="btn small" onClick={(e) => { e.stopPropagation(); setSlot(i, null); setActive(i); }}>×</button>}
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <input className={`input ${typedErr ? "invalid" : ""}`} value={typed} onChange={(e) => setTyped(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") applyTyped(); }}
          placeholder={`type into ${active === 0 ? "A" : "B"} — বাংলা / Banglish`} spellCheck={false} />
        <button className="btn" onClick={applyTyped} disabled={!typed.trim() || !!typedErr}>Set</button>
      </div>
      {typedErr && <div className="error">{typedErr}</div>}
      {err && <div className="error">{err}</div>}
      {res && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: 16 }}>
          {res.cos === null ? (
            <div className="error">Not in the model: {[res.a, res.b].filter((x) => !x.in_model).map(sideLabel).join(", ")}</div>
          ) : (
            <>
              <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                <span className="h-display" style={{ fontSize: 44, color: "#fff" }}>{res.cos.toFixed(3)}</span>
                <span className="mono faint">cosine · angle {res.angle_deg?.toFixed(1)}°</span>
              </div>
              <div className="note">{sideLabel(res.a)} ↔ {sideLabel(res.b)}</div>
              <div className="progress" style={{ height: 8, marginTop: 8 }}>
                <motion.div initial={{ width: 0 }} animate={{ width: `${pct * 100}%` }} transition={{ duration: 0.7 }} style={{ background: "linear-gradient(90deg, #7fb2ff, #f2bf4b, #ff3d5a)" }} />
              </div>
              <div className="mono faint" style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginTop: 3 }}><span>−1 opposite</span><span>0</span><span>+1 same direction</span></div>
              {[res.a, res.b].map((x, i) => x.kind_of_item === "sentence" ? (
                <div key={i} style={{ marginTop: 10 }}><span className="mono faint" style={{ fontSize: 10 }}>{i === 0 ? "A" : "B"} · {x.used} words used</span><Tokens tokens={x.tokens} /></div>
              ) : null)}
            </>
          )}
        </motion.div>
      )}
    </div>
  );
}

const STRENGTH_COLOR: Record<string, string> = { "very strong": "#7fe0b7", strong: "#9fe2c4", moderate: "#f2bf4b", weak: "#e8a06a", "not close": "#ff8a9b" };

function RelationsPanel({ rel, points, selected, onAdd, onAnalogy }: {
  rel: Relation[]; points: Point[]; selected: string | null; onAdd: (w: string) => void; onAnalogy: () => void;
}) {
  const label = (id: string) => points.find((p) => p.id === id)?.label ?? id;
  const list = [...rel].sort((a, b) => {
    const sa = selected && (a.source === selected || a.target === selected) ? 1 : 0;
    const sb = selected && (b.source === selected || b.target === selected) ? 1 : 0;
    return sb - sa || b.cos - a.cos;
  });
  if (!rel.length) return <p className="muted">Add at least two words or sentences: every pair of your items is related here by the merged model.</p>;
  return (
    <div>
      <div className="eyebrow">your items, pair by pair · full 900-d vectors</div>
      <p className="note" style={{ margin: "6px 0 12px" }}>
        The <b>similarity</b> (cosine) is one number for a pair, the same both ways. A <b>list position</b> is one-way: where B falls among A’s
        nearest words (#1 = closest of 273,531); the two positions of a pair differ because some words have more close neighbours than others.
        <b> Bridge words</b> are words the model puts close to both.
      </p>
      {list.map((r, i) => {
        const near = r.rank_ab ?? r.rank_ba;
        return (
          <motion.div key={r.source + r.target} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}
            style={{ padding: "10px 0", borderTop: "1px solid var(--line)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
              <span style={{ fontSize: 16 }}>{label(r.source)} <span className="faint">↔</span> {label(r.target)}</span>
              <span className="mono" style={{ fontSize: 18, color: "#8fd3ff" }}>{r.cos.toFixed(3)}</span>
            </div>
            <div className="progress" style={{ height: 5, margin: "6px 0" }}>
              <motion.div initial={{ width: 0 }} animate={{ width: `${Math.max(0, r.cos) * 100}%` }} transition={{ duration: 0.6 }} style={{ background: "#8fd3ff" }} />
            </div>
            <div className="note" style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {r.strength && <span className="verdict" style={{ background: "rgba(255,255,255,0.05)", color: STRENGTH_COLOR[r.strength] }}>{r.strength}</span>}
              <span>same score in both directions</span>
            </div>
            <div className="note" style={{ marginTop: 4 }} title="A rank is a position in one word's own list of nearest words. Lists differ in how crowded they are, so the two ranks of a pair differ even though the similarity is the same.">
              list position (one-way):{" "}
              {r.rank_ab !== null && <span className="mono">#{r.rank_ab.toLocaleString()} in {label(r.source)}’s list</span>}
              {r.rank_ab !== null && r.rank_ba !== null && " · "}
              {r.rank_ba !== null && <span className="mono">#{r.rank_ba.toLocaleString()} in {label(r.target)}’s list</span>}
            </div>
            <div className="chips" style={{ marginTop: 6 }}>
              <span className="note">bridge words:</span>
              {r.bridges.map((b) => (
                <button key={b.word} className="btn small" onClick={() => onAdd(b.word)} title={`cos ${b.cos_a} to ${label(r.source)}, ${b.cos_b} to ${label(r.target)} · click to add`}>
                  {b.word} <span className="mono faint" style={{ fontSize: 10 }}>{Math.min(b.cos_a, b.cos_b).toFixed(2)}</span>
                </button>
              ))}
            </div>
            {near !== null && near > 1000 && r.source.startsWith("w:") && r.target.startsWith("w:") && (
              <div className="note" style={{ marginTop: 6 }}>
                These words rarely share contexts, so the model does not place them close. A relation such as
                “রাজা is to রানী as পুরুষ is to মহিলা” is a direction, not a closeness —
                <button className="btn small" style={{ marginLeft: 6 }} onClick={onAnalogy}>try it in Analogy</button>
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}
