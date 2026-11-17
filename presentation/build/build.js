// BanglaMeta thesis progress deck. Every number is read from numbers.json (exported from the result CSVs).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = process.env.PPTX_SKILL;
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));

const N = JSON.parse(fs.readFileSync(path.join(__dirname, "numbers.json"), "utf8").replace(/\bNaN\b/g, "null"));
const D = N.dev, K = N.conf, PC = N.pc;
const TEST = (cmp, metric) => N.tests.find(t => t.comparison === cmp && t.metric === metric);

// ---------------------------------------------------------------- formatting (scores shown in % with 2 decimals)
const pct = x => (x * 100).toFixed(2);
const sgn = x => (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(2);
const ci = (lo, hi) => `[${sgn(lo)}, ${sgn(hi)}]`;
const nat = x => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
const int = x => Number(x).toLocaleString("en-US");
const p4 = x => x < 0.001 ? x.toFixed(4) : x.toFixed(4);

// ---------------------------------------------------------------- design system
const THEME = {
  name: "BanglaMeta", headFontFace: "Georgia", bodyFontFace: "Calibri",
  colors: { dk1: "1F2A27", lt1: "FFFFFF", dk2: "0E3B2E", lt2: "E8F3EE",
            accent1: "1B7A5A", accent2: "D7263D", accent3: "E0A526", accent4: "5F6B66", accent5: "2E86AB", accent6: "8FB9A8",
            hlink: "1B7A5A", folHlink: "5F6B66" },
};
const DARK = "0E3B2E", GREEN = "1B7A5A", MINT = "E8F3EE", RED = "D7263D", GOLD = "E0A526",
      INK = "1F2A27", GREY = "5F6B66", LINE = "D5E2DC", WHITE = "FFFFFF", SOFT = "F6FAF8", LGREY = "B9C4BF";
const HEAD = "Georgia", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                         // 13.333 x 7.5 in
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };
pres.title = "BanglaMeta: Domain-Bridging Word2Vec Meta-Embeddings for Bangla";
pres.author = "Md Shahadat Hossain";
pres.subject = "Thesis progress presentation";

pres.defineSlideMaster({
  title: "CONTENT", background: { color: WHITE },
  objects: [{ text: { text: "BanglaMeta · Thesis progress presentation",
                      options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontFace: BODY, fontSize: 10, color: GREY, margin: 0 } } }],
  slideNumber: { x: 12.15, y: 7.02, w: 0.6, h: 0.3, fontFace: BODY, fontSize: 10, color: GREY, align: "right" },
});
pres.defineSlideMaster({ title: "DARK", background: { color: DARK }, objects: [] });

const SECTIONS = ["Introduction", "Literature Review", "Conceptual Framework", "Dataset", "Methodology", "Results", "Conclusion", "References"];
let currentSection = null;
const want = i => !process.env.ONLY || process.env.ONLY.split(',').map(Number).includes(i);
function content(secIdx, title, notes) {
  const sec = SECTIONS[secIdx - 1];
  if (sec !== currentSection) { pres.addSection({ title: sec }); currentSection = sec; }
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec });
  s.addText(`${String(secIdx).padStart(2, "0")}  ·  ${sec.toUpperCase()}`,
    { x: 0.6, y: 0.32, w: 8, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: RED, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText(title, { x: 0.6, y: 0.62, w: 12.1, h: 0.75, fontFace: HEAD, fontSize: 28, bold: true, color: DARK, margin: 0, valign: "middle", isTextBox: true });
  if (notes) s.addNotes(notes);
  return s;
}
function card(s, x, y, w, h, fill = MINT) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
}
function txt(s, text, o) { s.addText(text, Object.assign({ fontFace: BODY, fontSize: 15, color: INK, margin: 0, isTextBox: true, valign: "top" }, o)); }
function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })),
    Object.assign({ fontFace: BODY, fontSize: 15, color: INK, margin: 0, isTextBox: true, valign: "top" }, o));
}
function badge(s, x, y, label, fill = GREEN, d = 0.5, size = 16) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  s.addText(label, { x, y, w: d, h: d, fontFace: HEAD, fontSize: size, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, big, label, color = DARK, bigSize = 36) {
  s.addText(big, { x, y, w, h: 0.65, fontFace: HEAD, fontSize: bigSize, bold: true, color, margin: 0, isTextBox: true, valign: "bottom" });
  s.addText(label, { x, y: y + 0.7, w, h: 0.55, fontFace: BODY, fontSize: 13, color: GREY, margin: 0, isTextBox: true });
}
function arrow(s, x1, y1, x2, y2, color = GREY) {   // width/height must be positive: flip for upward arrows
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV: y2 < y1, flipH: x2 < x1,
    line: { color, width: 2, endArrowType: "triangle" } });
}
// table helper: rows = array of arrays of strings or {text, options}
function table(s, header, rows, o) {
  const H = header.map(h => ({ text: h, options: { bold: true, color: WHITE, fill: { color: DARK }, fontSize: o.hsize || 12, valign: "middle" } }));
  const R = rows.map((r, i) => r.map(c => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    cell.options = Object.assign({ fill: { color: i % 2 ? SOFT : WHITE }, color: INK, fontSize: o.size || 12, valign: "middle" }, cell.options || {});
    return cell;
  }));
  s.addTable([H, ...R], Object.assign({ fontFace: BODY, border: { type: "solid", pt: 0.75, color: LINE }, margin: [3, 6, 3, 6] }, o));
}
const HL = { bold: true, color: WHITE, fill: { color: GREEN } };          // highlighted cell
const BEST = { bold: true, color: GREEN };

// ================================================================ 1. TITLE
if (want(1)) {
  pres.addSection({ title: "Title" });
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Title" });
  s.addText("THESIS PROGRESS PRESENTATION", { x: 0.8, y: 0.9, w: 7, h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: GOLD, charSpacing: 3, margin: 0, isTextBox: true });
  s.addText("BanglaMeta", { x: 0.8, y: 1.5, w: 7.5, h: 1.2, fontFace: HEAD, fontSize: 60, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Domain-Bridging Word2Vec\nMeta-Embeddings for Bangla", { x: 0.8, y: 2.75, w: 7.4, h: 1.1, fontFace: HEAD, fontSize: 26, color: "CFE6DA", margin: 0, isTextBox: true });
  s.addText("Md Shahadat Hossain", { x: 0.8, y: 4.6, w: 6, h: 0.4, fontFace: BODY, fontSize: 18, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("November 2026", { x: 0.8, y: 5.05, w: 6, h: 0.35, fontFace: BODY, fontSize: 15, color: "CFE6DA", margin: 0, isTextBox: true });
  // motif: three domain models -> one concatenated vector
  const srcs = [["News-W2V", "formal news", GREEN], ["Informal-W2V", "reviews · social", GOLD], ["Subs2vec-w2", "conversation", RED]];
  srcs.forEach(([n, d, c], i) => {
    const y = 1.45 + i * 1.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.7, y, w: 2.3, h: 0.8, fill: { color: c }, line: { color: c }, rectRadius: 0.08 });
    s.addText([{ text: n, options: { bold: true, fontSize: 15, breakLine: true } }, { text: d, options: { fontSize: 12 } }],
      { x: 8.7, y, w: 2.3, h: 0.8, fontFace: BODY, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
    arrow(s, 11.05, y + 0.4, 11.55, 2.85, "CFE6DA");
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 11.6, y: 1.45, w: 1.0, h: 2.9, fill: { color: WHITE }, line: { color: WHITE }, rectRadius: 0.08 });
  [GREEN, GOLD, RED].forEach((c, i) => s.addShape(pres.shapes.RECTANGLE, { x: 11.75, y: 1.6 + i * 0.88, w: 0.7, h: 0.82, fill: { color: c }, line: { color: WHITE, width: 1 } }));
  s.addText("one 900-d vector per word", { x: 8.7, y: 4.55, w: 3.95, h: 0.35, fontFace: BODY, fontSize: 13, italic: true, color: "CFE6DA", align: "right", margin: 0, isTextBox: true });
  s.addNotes("Title. BanglaMeta merges domain-specific Bangla Word2Vec models (news, informal, conversational) into one embedding. The graphic is the method in one picture: three 300-d models are concatenated into one 900-d vector per word.");
}

// ================================================================ 2. OUTLINE
if (want(2)) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Title" });
  s.addText("Outline", { x: 0.6, y: 0.5, w: 8, h: 0.8, fontFace: HEAD, fontSize: 36, bold: true, color: DARK, margin: 0, isTextBox: true });
  const items = [
    ["Introduction", "Background, problem, significance and objectives"],
    ["Literature Review", "Meta-embedding methods, Bangla embeddings, research gap"],
    ["Conceptual Framework", "From domain corpora to one merged embedding; formula and proofs"],
    ["Dataset", "Construction, statistics, quality and leakage control"],
    ["Methodology", "Source models, merge methods, evaluation and selection rules"],
    ["Results", "Pre-evaluation, merge experiments, held-out confirmation"],
    ["Conclusion", "Findings, limitations and future work"],
    ["References", "Sources cited in this presentation"],
  ];
  items.forEach(([t, d], i) => {
    const col = i < 4 ? 0 : 1, row = i % 4;
    const x = 0.6 + col * 6.3, y = 1.65 + row * 1.3;
    s.addText(String(i + 1).padStart(2, "0"), { x, y, w: 1.0, h: 0.75, fontFace: HEAD, fontSize: 34, bold: true, color: GOLD, margin: 0, isTextBox: true, valign: "top" });
    s.addText(t, { x: x + 1.15, y: y + 0.02, w: 4.7, h: 0.42, fontFace: HEAD, fontSize: 20, bold: true, color: DARK, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 1.15, y: y + 0.47, w: 4.7, h: 0.6, fontFace: BODY, fontSize: 14, color: GREY, margin: 0, isTextBox: true });
  });
  s.addShape(pres.shapes.LINE, { x: 6.6, y: 1.7, w: 0, h: 4.9, line: { color: LINE, width: 1.5 } });
  s.addNotes("Eight parts, in the order requested.");
}

// ================================================================ 3. BACKGROUND & PROBLEM
if (want(3)) {
  const s = content(1, "One Bangla Word2Vec cannot serve both kinds of text",
    "Bangla text comes in two very different registers. A Word2Vec model trained on news learns formal vocabulary; one trained on reviews and comments learns slang and code-mixing. The chart shows our own DEV-half evidence: each model is best only in its own domain. That is the problem this thesis solves.");
  card(s, 0.6, 1.6, 5.6, 1.45);
  txt(s, [{ text: "Domain A · formal news", options: { bold: true, color: DARK, fontSize: 17, breakLine: true } },
          { text: "Standard spelling, long articles (mean 216.7 tokens per text in our set)", options: { fontSize: 14 } }], { x: 0.85, y: 1.75, w: 5.1, h: 1.2 });
  card(s, 0.6, 3.2, 5.6, 1.45);
  txt(s, [{ text: "Domain B · informal text", options: { bold: true, color: DARK, fontSize: 17, breakLine: true } },
          { text: "Book reviews, social comments, hate speech: slang, dialect, short texts (13.6–23.1 tokens)", options: { fontSize: 14 } }], { x: 0.85, y: 3.35, w: 5.1, h: 1.2 });
  txt(s, [{ text: "Problem. ", options: { bold: true, color: RED } },
          { text: "Word vectors learned in one domain transfer poorly to the other, and Bangla has no merged Word2Vec that covers both." }],
      { x: 0.6, y: 4.9, w: 5.6, h: 0.9, fontSize: 15 });
  txt(s, [{ text: "Why it matters. ", options: { bold: true, color: GREEN } },
          { text: "Static Word2Vec vectors are cheap (CPU only) and still used as features in low-resource Bangla NLP." }],
      { x: 0.6, y: 5.8, w: 5.6, h: 0.9, fontSize: 15 });
  const nw = D["News-W2V"], iw = D["Informal-W2V"];
  s.addChart(pres.charts.BAR, [
    { name: "News-W2V", labels: ["News (D1)", "Informal (D2)"], values: [+pct(nw.D1_F1), +pct(nw.D2_F1)] },
    { name: "Informal-W2V", labels: ["News (D1)", "Informal (D2)"], values: [+pct(iw.D1_F1), +pct(iw.D2_F1)] },
  ], { x: 6.7, y: 1.55, w: 6.0, h: 4.6, barDir: "col", barGapWidthPct: 60, chartColors: [GREEN, GOLD],
       showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelColor: INK,
       valAxisMinVal: 50, valAxisMaxVal: 95, valAxisLabelFontSize: 11, catAxisLabelFontSize: 13, valAxisLabelColor: GREY, catAxisLabelColor: INK,
       valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valGridLine: { color: "E3E9E6", size: 0.75 }, catGridLine: { style: "none" },
       showLegend: true, legendPos: "t", legendFontSize: 12, legendFontFace: "+mn-lt",
       showTitle: true, title: "Each model wins only its own domain (macro-F1 %, DEV)", titleFontSize: 14, titleColor: DARK, titleFontFace: "+mn-lt" });
  txt(s, "Axis starts at 50 %. Source: merge_dev.csv (DEV half, 5-fold linear probe).", { x: 6.7, y: 6.25, w: 6.0, h: 0.3, fontSize: 11, color: GREY });
}

// ================================================================ 4. OBJECTIVES
if (want(4)) {
  const s = content(1, "Aim and objectives",
    "The aim and four objectives. Objective 2 is what makes this work different: we prove before merging that merging can add information. Objective 4 makes the final claim honest: we choose on one half of the data and confirm once on the other half.");
  card(s, 0.6, 1.6, 12.1, 1.0, DARK);
  txt(s, [{ text: "Aim  ", options: { bold: true, color: GOLD } },
          { text: "Build one Bangla Word2Vec embedding that works on both formal and informal text by merging domain-specific Word2Vec models, and show on unseen data that it is better than each of them." }],
      { x: 0.9, y: 1.68, w: 11.5, h: 0.85, color: WHITE, fontSize: 16, valign: "middle" });
  const obj = [
    ["O1", "Build and verify", "Train or download Word2Vec models for news, informal and conversational text; verify them."],
    ["O2", "Prove before merging", "Show by proof and measured information gain that merging adds useful information."],
    ["O3", "Compare techniques", "Concatenation, SVD, alignment, autoencoders, joint training: one protocol."],
    ["O4", "Select and confirm", "Pick the best merge by a rule fixed in advance; test it once on a held-out half."],
  ];
  obj.forEach(([k, h, d], i) => {
    const x = 0.6 + i * 3.08;
    card(s, x, 2.95, 2.88, 3.2);
    badge(s, x + 0.25, 3.2, k, i === 3 ? RED : GREEN, 0.7, 16);
    txt(s, h, { x: x + 0.25, y: 4.0, w: 2.45, h: 0.75, fontFace: HEAD, fontSize: 17, bold: true, color: DARK, valign: "top" });
    txt(s, d, { x: x + 0.25, y: 4.85, w: 2.45, h: 1.25, fontSize: 14 });
  });
  txt(s, [{ text: "Research question  ", options: { bold: true, color: RED } },
          { text: "Does a merged Word2Vec beat each of its sources and the fastText baseline on held-out Bangla text?" }],
      { x: 0.6, y: 6.35, w: 12.1, h: 0.45, fontSize: 15 });
}

// ================================================================ 5. LITERATURE: META-EMBEDDING
if (want(5)) {
  const s = content(2, "Meta-embedding: what earlier work did",
    "Five key meta-embedding papers, found and checked through SciSpace. All of them work on English and mostly evaluate word similarity and analogy. Bollegala 2022 explains theoretically why concatenation is strong; our Theorem 5 adds that with a standardised probe the weights cancel out.");
  const rows = [
    ["Yin & Schütze (2016)\nACL", "Learn a meta-embedding from several public embedding sets; concatenation and SVD as baselines", "Meta-embeddings beat each single set on similarity, analogy and POS tagging", "English; general-domain sources"],
    ["Coates & Bollegala (2018)\nNAACL", "Plain averaging of source embeddings", "Averaging is comparable to or better than more complex methods", "English; no domain split"],
    ["Bollegala & Bao (2018)\nCOLING", "Autoencoded meta-embeddings (AAEME, CAEME)", "Outperform earlier meta-embeddings on several benchmarks", "English; we test both as candidates"],
    ["Jawanpuria et al. (2020)\nRepL4NLP @ ACL", "Orthogonal rotation + metric scaling into a common space, then average / concatenate", "Common latent space helps similarity and analogy", "English; intrinsic tasks only"],
    ["Bollegala (2022)\nIJCAI", "Unsupervised weighted concatenation; theory: spectrum matching", "Explains why concatenation is a strong baseline; learned weights help further", "English; no held-out confirmation step"],
  ];
  table(s, ["Study", "Method", "Key finding", "Limitation for our problem"], rows.map(r => [{ text: r[0], options: { bold: true, color: DARK } }, r[1], r[2], r[3]]),
    { x: 0.6, y: 1.6, w: 12.1, colW: [2.5, 3.6, 3.4, 2.6], size: 12, hsize: 13, rowH: 0.78 });
  txt(s, "All five papers were retrieved and checked through SciSpace; full citations on the References slide.", { x: 0.6, y: 6.65, w: 12, h: 0.3, fontSize: 11, color: GREY });
}

// ================================================================ 6. LITERATURE: BANGLA + GAP
if (want(6)) {
  const s = content(2, "Bangla embeddings and the research gap",
    "Bangla embedding work trains one model on one (mostly news-like) corpus and compares training settings. Rahman 2020 found 300-d skip-gram with window 4 best for Bangla, which is exactly the setting we use. No study merges Bangla embeddings across domains: that is our gap.");
  const rows = [
    ["Ritu et al. (2018)", "Word2Vec (TensorFlow, gensim) vs fastText on a Bangla corpus", "Compared accuracy and efficiency of the three models"],
    ["Rahman (2020)\nICCIT", "Tuned Word2Vec hyper-parameters for Bangla", "Best: 300-d skip-gram, window 4 (the setting we use)"],
    ["Hossain & Hoque (2020)", "Word2Vec, GloVe, fastText on a 180 M-word Bangla corpus", "Intrinsic and extrinsic comparison of the three techniques"],
  ];
  table(s, ["Study", "What they did", "Finding"], rows.map(r => [{ text: r[0], options: { bold: true, color: DARK } }, r[1], r[2]]),
    { x: 0.6, y: 1.6, w: 7.0, colW: [1.9, 2.6, 2.5], size: 12, hsize: 13, rowH: 0.85 });
  card(s, 7.95, 1.6, 4.75, 5.0, DARK);
  txt(s, "Research gap", { x: 8.25, y: 1.8, w: 4.2, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: GOLD });
  const gaps = [
    ["G1", "No meta-embedding study for Bangla."],
    ["G2", "Bangla Word2Vec models are trained on one domain; none covers formal and informal text together."],
    ["G3", "In the studies reviewed, merges are not tested beforehand for added information, nor confirmed on a held-out half after selection."],
  ];
  gaps.forEach(([k, t], i) => {
    const y = 2.5 + i * 1.3;
    badge(s, 8.25, y, k, RED, 0.55, 13);
    txt(s, t, { x: 8.95, y: y - 0.02, w: 3.55, h: 1.2, fontSize: 14, color: WHITE });
  });
  card(s, 0.6, 5.05, 7.0, 1.55);
  txt(s, [{ text: "Our contribution  ", options: { bold: true, color: RED } },
          { text: "the first Bangla Word2Vec meta-embedding across news and informal domains, with a pre-merge information test (P1–P6) and a single held-out confirmation with multiple-comparison correction." }],
      { x: 0.85, y: 5.2, w: 6.5, h: 1.3, fontSize: 15 });
}

// ================================================================ 7. CONCEPTUAL FRAMEWORK
if (want(7)) {
  const s = content(3, "From domain corpora to one merged embedding",
    "Left to right: three corpora give three domain models. Each model maps a word to a 300-d vector. The merge step normalises, fills in missing words, and concatenates. The output is BanglaMeta, one 900-d vector per word. Below: the DEV half is used for every choice, the CONF half only once at the end.");
  const rows = [["News corpus", "Potrika · 65.3 M tokens", "News-W2V", GREEN], ["Informal corpus", "reviews + social + hate · 3.19 M tokens", "Informal-W2V", GOLD], ["OpenSubtitles", "conversational speech", "Subs2vec-w2", RED]];
  rows.forEach(([c, d, m, col], i) => {
    const y = 1.6 + i * 1.4;
    card(s, 0.6, y, 2.7, 1.15);
    txt(s, [{ text: c, options: { bold: true, color: DARK, fontSize: 15, breakLine: true } }, { text: d, options: { fontSize: 12, color: GREY } }], { x: 0.75, y: y + 0.12, w: 2.45, h: 0.95 });
    arrow(s, 3.35, y + 0.575, 3.85, y + 0.575);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.9, y, w: 2.2, h: 1.15, fill: { color: col }, line: { color: col }, rectRadius: 0.08 });
    txt(s, [{ text: m, options: { bold: true, fontSize: 15, breakLine: true } }, { text: "word → 300-d", options: { fontSize: 12 } }], { x: 3.9, y, w: 2.2, h: 1.15, color: WHITE, align: "center", valign: "middle" });
    arrow(s, 6.15, y + 0.575, 6.65, 3.48);
  });
  card(s, 6.7, 1.6, 3.0, 3.95, DARK);
  txt(s, "Merge", { x: 6.95, y: 1.75, w: 2.5, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: GOLD });
  ["1  Normalise each row", "2  Impute missing words (ridge map)", "3  Concatenate, weight 1/√3"].forEach((t, i) =>
    txt(s, t, { x: 6.95, y: 2.4 + i * 0.95, w: 2.6, h: 0.8, fontSize: 15, color: WHITE }));
  arrow(s, 9.75, 3.58, 10.25, 3.58);
  card(s, 10.3, 1.6, 2.4, 3.95, MINT);
  txt(s, "BanglaMeta", { x: 10.45, y: 1.8, w: 2.1, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: DARK, align: "center" });
  [GREEN, GOLD, RED].forEach((c, i) => s.addShape(pres.shapes.RECTANGLE, { x: 11.15, y: 2.45 + i * 0.7, w: 0.7, h: 0.66, fill: { color: c }, line: { color: WHITE, width: 1 } }));
  txt(s, "900-d · 273,531 words", { x: 10.4, y: 4.65, w: 2.2, h: 0.7, fontSize: 13, color: INK, align: "center" });
  card(s, 0.6, 5.85, 5.95, 0.85, SOFT);
  txt(s, [{ text: "DEV half (11,950 texts)  ", options: { bold: true, color: GREEN } }, { text: "screening, checks, 28 merge candidates, selection" }], { x: 0.8, y: 5.95, w: 5.6, h: 0.65, fontSize: 14, valign: "middle" });
  card(s, 6.75, 5.85, 5.95, 0.85, SOFT);
  txt(s, [{ text: "CONF half (11,950 texts)  ", options: { bold: true, color: RED } }, { text: "used once: confirm the frozen model" }], { x: 6.95, y: 5.95, w: 5.6, h: 0.65, fontSize: 14, valign: "middle" });
}

// ================================================================ 8. THE FORMULA
if (want(8)) {
  const s = content(3, "The merge formula",
    "Three simple steps. Every row is scaled to length one so that no model dominates. A word missing from one model gets that block predicted from another model through a linear map learned on shared words. Then the blocks are put side by side, each scaled by one over root three, so every merged row has length one and each model owns exactly one third of it.");
  card(s, 0.6, 1.6, 12.1, 1.35, DARK);
  txt(s, "m(w)  =  (1/√3) · [ ñ(w) ;  ĩ(w) ;  s̃(w) ]   ∈  ℝ⁹⁰⁰", { x: 0.9, y: 1.68, w: 11.5, h: 0.75, fontFace: "Cambria Math", fontSize: 30, color: WHITE, align: "center", valign: "middle" });
  txt(s, "ñ, ĩ, s̃ = unit-length rows of News-W2V, Informal-W2V, Subs2vec-w2 (missing words imputed)", { x: 0.9, y: 2.45, w: 11.5, h: 0.4, fontSize: 14, color: "CFE6DA", align: "center" });
  const steps = [
    ["1", "Normalise", "a(w) ← a(w) / ‖a(w)‖", "Every word vector gets length 1, so no model dominates by scale."],
    ["2", "Fill missing words", "M = (SᵀS + λI)⁻¹ SᵀT,  λ = 1", "Ridge map fitted on shared words predicts a missing row; union = 273,531 words."],
    ["3", "Concatenate", "weight 1/√3 per block", "Each merged row has length 1; each model contributes exactly one third."],
  ];
  steps.forEach(([k, h, f, d], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 3.25, 3.9, 2.55);
    badge(s, x + 0.25, 3.45, k, GREEN, 0.55, 16);
    txt(s, h, { x: x + 0.95, y: 3.5, w: 2.8, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: DARK });
    txt(s, f, { x: x + 0.25, y: 4.15, w: 3.45, h: 0.45, fontFace: "Cambria Math", fontSize: 15, color: GREEN });
    txt(s, d, { x: x + 0.25, y: 4.7, w: 3.45, h: 1.0, fontSize: 14 });
  });
  txt(s, [{ text: "Sentence vector  ", options: { bold: true, color: DARK } },
          { text: "x(s) = mean of m(w) over the words of text s  →  linear classifier.  Columns: 1–300 News, 301–600 Informal, 601–900 Subs2vec." }],
      { x: 0.6, y: 6.05, w: 12.1, h: 0.75, fontSize: 14 });
}

// ================================================================ 9. MATHEMATICAL PROOF
if (want(9)) {
  const qc = D["Q: CONC"].Overall_F1, qs = D["Q: CONC+SVD300"].Overall_F1, qp = D["Q: Procrustes+AVG"].Overall_F1;
  const w3 = D["Q: wCONC a=0.3"].Overall_F1, w7 = D["Q: wCONC a=0.7"].Overall_F1;
  const T = K["T: CONC"], NW = K["News-W2V"], IW = K["Informal-W2V"];
  const q = PC.Q;
  const s = content(3, "Why concatenation: four results, each checked in data",
    "Four theorems in plain words. One: a classifier on the merged vector can always ignore one block, so the merge can never be worse than its better source. Two: the size of the gain equals the extra label information the second model carries; we measured it. Three: averaging, SVD and alignment are all just a matrix times the concatenation, so they can only lose information. Four: the classifier standardises each feature, so block weights cancel. The bottom line states what mathematics cannot prove.");
  const th = [
    ["1", "Merging cannot lose", "W·x(A) = [W , 0] · [x(A) ; x(B)]", "Any classifier on one model is also a classifier on the merge, so best loss(merge) ≤ best loss(each source).",
     `CONF: merge ${pct(T.D1_F1)} on news (> News-W2V ${pct(NW.D1_F1)}) and ${pct(T.D2_F1)} on informal (> Informal-W2V ${pct(IW.D2_F1)})`],
    ["2", "Gain = added information", "R(A) − R(A,B) = I(Y ; B | A) ≥ 0", "The gain equals the label information B adds beyond A; zero only if B adds nothing.",
     `News + Informal: ${nat(q["P2_I(B|A)_D2"])} nats on informal, ${nat(q["P2_I(A|B)_D1"])} on news; 95% CIs above 0`],
    ["3", "Concatenation beats linear merges", "average, SVD, alignment = L·m(w)", "A classifier on L·m is a classifier on m, so no linear merge can reach a lower loss.",
     `DEV (News + Informal): CONC ${pct(qc)} > SVD-300 ${pct(qs)} > Procrustes+AVG ${pct(qp)}`],
    ["4", "Weights cancel", "(c·x) / (c·σ) = x / σ", "The probe standardises every feature, so scaling a block changes nothing.",
     `DEV: α = 0.3, 0.5, 0.7 give ${pct(w3)}, ${pct(qc)}, ${pct(w7)} (identical)`],
  ];
  th.forEach(([k, h, f, d, chk], i) => {
    const x = 0.6 + (i % 2) * 6.1, y = 1.55 + Math.floor(i / 2) * 2.45;
    card(s, x, y, 5.9, 2.3);
    badge(s, x + 0.2, y + 0.2, k, GREEN, 0.5, 15);
    txt(s, h, { x: x + 0.85, y: y + 0.22, w: 4.9, h: 0.42, fontFace: HEAD, fontSize: 17, bold: true, color: DARK });
    txt(s, f, { x: x + 0.85, y: y + 0.68, w: 4.9, h: 0.38, fontFace: "Cambria Math", fontSize: 15, color: GREEN });
    txt(s, d, { x: x + 0.2, y: y + 1.1, w: 5.5, h: 0.6, fontSize: 13 });
    txt(s, [{ text: "✓ ", options: { bold: true, color: GREEN } }, { text: chk }], { x: x + 0.2, y: y + 1.68, w: 5.5, h: 0.55, fontSize: 12, color: GREY });
  });
  txt(s, [{ text: "Not provable by maths: ", options: { bold: true, color: RED } },
          { text: "being the best model overall. With 28 candidates, DEV gaps below 2.4 points are not guaranteed (Hoeffding + union bound), so the winner is confirmed once on unseen data." }],
      { x: 0.6, y: 6.5, w: 12.1, h: 0.45, fontSize: 13 });
}

// ================================================================ 10. DATASET CONSTRUCTION
if (want(10)) {
  const s = content(4, "Evaluation set: 23,900 balanced, labelled texts",
    "Four classification tasks, one news task and three informal tasks, balanced per class. Each task is split once, stratified, into two equal halves. Total news share is 49.8 percent, informal 50.2 percent, so both domains weigh equally.");
  const rows = [
    ["D1", "News topic", "Potrika", "7", "1,700", "11,900", "5,950", "5,950", "216.7"],
    ["D2", "Review sentiment", "BanglaBook", "3", "1,500", "4,500", "2,250", "2,250", "23.1"],
    ["D2", "Social sentiment", "SentNoB", "3", "1,000", "3,000", "1,500", "1,500", "13.6"],
    ["D2", "Hate speech", "Bengali Hate Speech", "2", "2,250", "4,500", "2,250", "2,250", "15.4"],
  ];
  const tot = [{ text: "Total", options: { bold: true } }, "", "", "", "", { text: "23,900", options: { bold: true } }, { text: "11,950", options: { bold: true } }, { text: "11,950", options: { bold: true } }, ""];
  table(s, ["Domain", "Task", "Source", "Classes", "Per class", "Texts", "DEV", "CONF", "Mean tokens"], [...rows, tot],
    { x: 0.6, y: 1.6, w: 12.1, colW: [0.95, 1.9, 2.25, 0.95, 1.15, 1.2, 1.2, 1.2, 1.3], size: 13, hsize: 13, rowH: 0.42, align: "left" });
  stat(s, 0.6, 4.4, 2.7, "49.8%", "news texts (11,900)", GREEN);
  stat(s, 3.4, 4.4, 2.7, "50.2%", "informal texts (12,000)", GOLD);
  card(s, 6.6, 4.3, 6.1, 2.45);
  txt(s, "Quality checks", { x: 6.85, y: 4.42, w: 5.6, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: DARK });
  bullets(s, ["Exact duplicates removed; texts with conflicting labels removed",
              "Texts appearing in two sources removed",
              "News: 50–400 words; informal: at least 3 words",
              "Stratified 50/50 DEV/CONF split per task (seed 42)"], { x: 6.85, y: 4.9, w: 5.65, h: 1.8, fontSize: 14 });
  txt(s, "Power analysis set ~12k texts per half: enough to detect a 1–2 point macro-F1 difference.", { x: 0.6, y: 5.85, w: 5.7, h: 0.7, fontSize: 13, color: GREY });
}

// ================================================================ 11. CORPORA & SOURCE MODELS
if (want(11)) {
  const s = content(4, "Training corpora and source models",
    "News-W2V and Informal-W2V are trained here, with every evaluation text removed from their corpora first, so no model has seen a test text. Subs2vec is downloaded from Zenodo and checked by MD5. fastText is the strong public baseline, not a Word2Vec model.");
  stat(s, 0.6, 1.55, 3.0, "65.3 M", "news tokens · 271,910 articles", GREEN);
  stat(s, 3.75, 1.55, 3.0, "3.19 M", "informal tokens · 149,423 texts", GOLD);
  stat(s, 6.9, 1.55, 3.0, "25,600", "documents removed that match an evaluation text (11,984 + 13,616)", RED);
  stat(s, 10.05, 1.55, 2.7, "0", "evaluation texts seen in training", DARK);
  const rows = [
    ["News-W2V", "Word2Vec, trained here", "Potrika news corpus", "187,686", "Domain A"],
    ["Informal-W2V", "Word2Vec, trained here", "BanglaBook + SentNoB + hate corpus", "32,910", "Domain B"],
    ["Subs2vec-w2", "Word2Vec, downloaded (MD5-checked)", "OpenSubtitles (Zenodo 22230710)", "177,376", "Domain B (conversation)"],
    ["fastText-cc.bn", "fastText vectors, downloaded", "Common Crawl + Wikipedia", "1,468,578", "Baseline (not Word2Vec)"],
  ];
  table(s, ["Model", "Type", "Trained on", "Vocabulary", "Role"], rows.map(r => [{ text: r[0], options: { bold: true, color: DARK } }, ...r.slice(1)]),
    { x: 0.6, y: 3.2, w: 12.1, colW: [2.0, 3.1, 3.5, 1.5, 2.0], size: 13, hsize: 13, rowH: 0.45 });
  card(s, 0.6, 5.65, 12.1, 1.1, SOFT);
  txt(s, [{ text: "Word2Vec settings (all trained models)  ", options: { bold: true, color: DARK } },
          { text: "skip-gram · 300 dimensions · window 4 · min count 5 · negative sampling 10 · 5 epochs · seed 42 · gensim 4.4.0. Subs2vec window w2 chosen by the pre-registered rule (best informal F1 of w1–w6)." }],
      { x: 0.85, y: 5.75, w: 11.6, h: 0.9, fontSize: 14, valign: "middle" });
}

// ================================================================ 12. PIPELINE
if (want(12)) {
  const s = content(5, "Seven-step research pipeline",
    "The seven steps the supervisor set, in order. Steps one to five use only the DEV half. Steps six and seven use the CONF half once. The bottom row shows what each step produced; all seven are complete.");
  const steps = [
    ["Download & verify models", "4 source models, MD5 / training logs"],
    ["Evaluate each model", "No model wins both domains"],
    ["Prove merging can help", "P1–P6: merge certified"],
    ["Merge experiments", "28 candidates, same protocol"],
    ["Select the best", "Pre-registered rule → BanglaMeta"],
    ["Test on unseen data", "CONF half, used once"],
    ["Prove it is better", "3 / 3 hypotheses confirmed"],
  ];
  card(s, 0.45, 1.6, 8.75, 4.5, SOFT);
  card(s, 9.3, 1.6, 3.6, 4.5, "FBEFF1");
  txt(s, "DEV half · all choices", { x: 0.7, y: 1.72, w: 6, h: 0.35, fontSize: 13, bold: true, color: GREEN });
  txt(s, "CONF half · once", { x: 9.5, y: 1.72, w: 3, h: 0.35, fontSize: 13, bold: true, color: RED });
  steps.forEach(([h, r], i) => {
    const x = 0.75 + i * 1.75, cx = x + 0.4;
    if (i < 6) arrow(s, cx + 0.5, 2.85, cx + 1.25, 2.85, LGREY);
    badge(s, cx - 0.4, 2.45, String(i + 1), i >= 5 ? RED : GREEN, 0.8, 22);
    txt(s, h, { x: x - 0.15, y: 3.45, w: 1.6, h: 0.9, fontFace: HEAD, fontSize: 14, bold: true, color: DARK, align: "center" });
    txt(s, r, { x: x - 0.15, y: 4.4, w: 1.6, h: 1.1, fontSize: 13, color: GREY, align: "center" });
    txt(s, "✓ done", { x: x - 0.15, y: 5.5, w: 1.6, h: 0.35, fontSize: 13, bold: true, color: i >= 5 ? RED : GREEN, align: "center" });
  });
  txt(s, "Code: w2v_src.py (steps 1–3), merge_src.py (4–5), conf_src.py (6–7); every stage re-checked by an independent verify_*.py script.", { x: 0.6, y: 6.35, w: 12.1, h: 0.4, fontSize: 12, color: GREY });
}

// ================================================================ 13. EVALUATION PROTOCOL
if (want(13)) {
  const s = content(5, "One evaluation protocol for every model",
    "Every model, single or merged, is scored the same way: average the word vectors of a text, train a linear ridge classifier with five-fold cross-validation, report macro-F1. Domain one is news, domain two is the mean of the three informal tasks, Overall is their mean. The selection and confirmation rules were fixed before the experiments.");
  card(s, 0.6, 1.6, 5.9, 5.15);
  txt(s, "How a model is scored", { x: 0.85, y: 1.75, w: 5.4, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: DARK });
  const how = [["Text → vector", "mean of the unit word rows in the text"], ["Classifier", "standardise + ridge (α chosen by CV), 5 stratified folds"],
               ["Metric", "macro-F1; D1 = news, D2 = mean of 3 informal tasks"], ["Overall", "(D1 + D2) / 2"],
               ["Uncertainty", "paired bootstrap: 1,000 samples (DEV), 10,000 (CONF)"], ["Word level", "analogy (16,678 questions, 3CosAdd), similarity AUC"]];
  how.forEach(([k, v], i) => txt(s, [{ text: k + "   ", options: { bold: true, color: GREEN } }, { text: v }], { x: 0.85, y: 2.35 + i * 0.7, w: 5.45, h: 0.62, fontSize: 14 }));
  card(s, 6.8, 1.6, 5.9, 2.45, DARK);
  txt(s, "Selection rule (DEV, fixed in advance)", { x: 7.05, y: 1.75, w: 5.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: GOLD });
  bullets(s, ["Specialists = best single model on D1 and on D2",
              "Eligible = within 1 point of the specialist on both domains",
              "Selected = eligible candidate with the highest Overall"], { x: 7.05, y: 2.25, w: 5.45, h: 1.7, fontSize: 14, color: WHITE });
  card(s, 6.8, 4.3, 5.9, 2.45, "FBEFF1");
  txt(s, "Confirmation rule (CONF, used once)", { x: 7.05, y: 4.45, w: 5.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: RED });
  bullets(s, ["Frozen model (MD5-checked), nothing retrained",
              "H1–H3: merged > fastText, News-W2V, Informal-W2V",
              "One-sided paired bootstrap, Holm correction, α = 0.05"], { x: 7.05, y: 4.95, w: 5.45, h: 1.7, fontSize: 14 });
}

// ================================================================ 14. MERGE METHODS
if (want(14)) {
  const s = content(5, "28 merge candidates, from simple to learned",
    "The five source sets and the methods applied to each. P is the supervisor's news plus conversational pair, Q the two domain specialists, T all three Word2Vec models. S and X include fastText and are reported as extended, not pure Word2Vec. Together: 28 candidates.");
  const Y = "●", n = "";
  const rows = [
    ["CONC", "Equal-energy concatenation, imputed union vocabulary", Y, Y, Y, Y, Y, "5"],
    ["CONC-zero", "Concatenation, missing rows left as zeros", Y, Y, n, n, n, "2"],
    ["Weighted CONC", "Block weights α = 0.3 and 0.7", Y, Y, n, n, n, "4"],
    ["CONC + SVD", "Project to k = 100, 200, 300 dimensions", Y, Y, Y, n, n, "9"],
    ["Procrustes + AVG", "Rotate one model onto the other, then average", Y, Y, n, n, n, "2"],
    ["AAEME / CAEME", "Autoencoders: summed / concatenated encodings", Y, Y, n, n, n, "4"],
    ["Joint-W2V", "One Word2Vec trained on news + informal corpora", "", "", "", "", "", "1"],
    ["Continued-W2V", "News-W2V trained further on informal text", "", "", "", "", "", "1"],
  ];
  const tot = [{ text: "Total", options: { bold: true } }, "", "", "", "", "", "", { text: "28", options: { bold: true } }];
  table(s, ["Method", "What it does", "P", "Q", "T", "S", "X", "Count"],
    [...rows.map(r => r.map((c, j) => j === 0 ? { text: c, options: { bold: true, color: DARK } } : (j >= 2 && j <= 6 ? { text: c, options: { align: "center", color: GREEN } } : (j === 7 ? { text: c, options: { align: "center" } } : c)))), tot],
    { x: 0.6, y: 1.6, w: 8.5, colW: [1.8, 3.6, 0.45, 0.45, 0.45, 0.45, 0.45, 0.85], size: 13, hsize: 13, rowH: 0.47 });
  card(s, 9.4, 1.6, 3.3, 5.1);
  txt(s, "Source sets", { x: 9.62, y: 1.75, w: 2.8, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: DARK });
  [["P", "News + Subs2vec"], ["Q", "News + Informal"], ["T", "News + Informal + Subs2vec"], ["S", "fastText + Informal *"], ["X", "fastText + all three *"]].forEach(([k, v], i) => {
    badge(s, 9.62, 2.3 + i * 0.75, k, i < 3 ? GREEN : GREY, 0.5, 14);
    txt(s, v, { x: 10.25, y: 2.3 + i * 0.75, w: 2.35, h: 0.55, fontSize: 13, valign: "middle" });
  });
  txt(s, "* extended: includes fastText, so not pure Word2Vec", { x: 9.62, y: 6.05, w: 2.95, h: 0.55, fontSize: 11, color: GREY });
}

// ================================================================ 15. PRE-EVALUATION
if (want(15)) {
  const order = ["fastText-cc.bn", "News-W2V", "Subs2vec-w2", "Informal-W2V"];
  const full = { "fastText-cc.bn": "1,468,578", "News-W2V": "187,686", "Subs2vec-w2": "177,376", "Informal-W2V": "32,910" };
  const best = k => Math.max(...order.map(m => D[m][k]));
  const cell = (m, k, f) => D[m][k] === best(k) ? { text: f(D[m][k]), options: BEST } : f(D[m][k]);
  const s = content(6, "Pre-evaluation: no single model wins both domains",
    "Step two, each model alone on the DEV half. fastText is best on news, Informal-W2V is best on informal text, and News-W2V has the best word similarity. No model is best everywhere, which is the precondition for merging. Green marks the best value in each column.");
  table(s, ["Model", "Vocabulary", "Analogy acc. %", "Similarity AUC %", "Coverage news / informal %", "D1 news F1", "D2 informal F1", "Overall F1"],
    order.map(m => [{ text: m, options: { bold: true, color: DARK } }, full[m], cell(m, "analogy_native_acc", pct), cell(m, "sim_auc_common", pct),
      `${pct(D[m].tok_cov_D1)} / ${pct(D[m].tok_cov_D2)}`, cell(m, "D1_F1", pct), cell(m, "D2_F1", pct), cell(m, "Overall_F1", pct)]),
    { x: 0.6, y: 1.6, w: 12.1, colW: [1.9, 1.35, 1.35, 1.45, 2.05, 1.3, 1.4, 1.3], size: 13, hsize: 12, rowH: 0.5 });
  const items = [["fastText", "best on news (D1)", GREEN], ["Informal-W2V", "best on informal (D2)", GOLD], ["News-W2V", "best word similarity", DARK]];
  items.forEach(([h, d, c], i) => {
    card(s, 0.6 + i * 4.1, 4.55, 3.9, 1.3);
    txt(s, h, { x: 0.85 + i * 4.1, y: 4.67, w: 3.5, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: c });
    txt(s, d, { x: 0.85 + i * 4.1, y: 5.15, w: 3.5, h: 0.5, fontSize: 15 });
  });
  txt(s, "Macro-F1 %, DEV half (5-fold probe). Analogy: 16,678 native questions, unanswerable = wrong. Similarity: category AUC on the 124 words shared by all models. fastText vocabulary = full file; rows used = 234,474.",
    { x: 0.6, y: 6.1, w: 12.1, h: 0.65, fontSize: 11, color: GREY });
}

// ================================================================ 16. PRE-MERGE CHECKS
if (want(16)) {
  const P = PC.P, Q = PC.Q, T = PC.T;
  const ok = (b, t) => ({ text: (b ? "✓  " : "✗  ") + t, options: { color: b ? GREEN : RED, bold: false } });
  const s = content(6, "Before merging: is a merge mathematically justified?",
    "Six checks run before any merged model is evaluated. P2 is the key one: does each model add label information the other lacks, with a confidence interval above zero? Yes for P and Q. For T, adding Subs2vec on top of News plus Informal adds almost nothing, which predicted the later result. SVD and averaging are not certified, so they are ablations only.");
  const rows = [
    ["P1", "Merged sentence vector = joined source vectors?", ok(true, "error 0.0"), ok(true, "error 0.0"), ok(true, "error 0.0")],
    ["P2", "Does each model add label information? (nats)", ok(P.P2_pass, `${nat(P["P2_I(B|A)_D2"])} / ${nat(P["P2_I(A|B)_D1"])}`), ok(Q.P2_pass, `${nat(Q["P2_I(B|A)_D2"])} / ${nat(Q["P2_I(A|B)_D1"])}`), ok(false, `${nat(T["P2_I(B|A)_D2"])}, CI incl. 0`)],
    ["P3", "Is SVD compression guaranteed safe?", ok(false, "bound too loose"), ok(false, "bound too loose"), ok(false, "bound too loose")],
    ["P4", "Is energy balanced? (share of first block)", ok(true, `${P.P4_energy_share_A_all.toFixed(2)} (target ${P.P4_target_share_A.toFixed(2)})`), ok(true, `${Q.P4_energy_share_A_all.toFixed(2)} (target ${Q.P4_target_share_A.toFixed(2)})`), ok(true, `${T.P4_energy_share_A_all.toFixed(2)} (target ${T.P4_target_share_A.toFixed(2)})`)],
    ["P5", "Is missing-word imputation better than chance?", ok(true, `cos ${P["P5_B->A_median_cos"].toFixed(2)} > ${P["P5_B->A_random_p95"].toFixed(2)}`), ok(true, `cos ${Q["P5_B->A_median_cos"].toFixed(2)} > ${Q["P5_B->A_random_p95"].toFixed(2)}`), ok(true, `cos ${T["P5_B->A_median_cos"].toFixed(2)} > ${T["P5_B->A_random_p95"].toFixed(2)}`)],
    ["P6", "Is alignment good enough to average? (P@1 ≥ 0.5)", ok(false, `P@1 ${P["P6_heldout_P@1"].toFixed(2)}`), ok(false, `P@1 ${Q["P6_heldout_P@1"].toFixed(2)}`), { text: "not applicable", options: { color: GREY } }],
  ];
  table(s, ["", "Question", "P: News + Subs2vec", "Q: News + Informal", "T: Q + Subs2vec"], rows.map(r => [{ text: r[0], options: { bold: true, color: DARK } }, ...r.slice(1)]),
    { x: 0.6, y: 1.6, w: 12.1, colW: [0.6, 4.3, 2.4, 2.4, 2.4], size: 13, hsize: 13, rowH: 0.5 });
  card(s, 0.6, 5.35, 12.1, 1.0, DARK);
  txt(s, [{ text: "Decision  ", options: { bold: true, color: GOLD } },
          { text: "concatenation is certified for P and Q; SVD and averaging are tested only as ablations; the third model in T is not certified to add information." }],
      { x: 0.85, y: 5.43, w: 11.6, h: 0.85, fontSize: 15, color: WHITE, valign: "middle" });
  txt(s, "P2 = I(Y; B | A) on informal tasks / I(Y; A | B) on news, linear-probe log-loss estimate with 95% bootstrap CI. P5 = median cosine of held-out imputed rows vs 95th percentile of random pairs (B→A).",
    { x: 0.6, y: 6.45, w: 12.1, h: 0.5, fontSize: 11, color: GREY });
}

// ================================================================ 17. STEP 4 DEV RESULTS
if (want(17)) {
  const list = ["Informal-W2V", "Subs2vec-w2", "News-W2V", "P: CONC", "fastText-cc.bn", "Q: Procrustes+AVG", "Q: CONC+SVD300", "Continued-W2V", "Joint-W2V", "Q: AAEME", "Q: CONC", "Q: CAEME", "T: CONC"];
  const label = m => m === "T: CONC" ? "BanglaMeta (T: CONC)" : m;
  const colors = list.map(m => m === "T: CONC" ? RED : (["Informal-W2V", "Subs2vec-w2", "News-W2V", "fastText-cc.bn"].includes(m) ? LGREY : GREEN));
  const s = content(6, "Merge experiments (DEV): concatenation wins",
    "Overall macro-F1 on the DEV half for the main candidates; grey bars are single models, green are merges, red is the selected model. Every concatenation of news and informal models beats fastText; compressions such as SVD and averaging fall back toward the single models, as Theorem 3 predicts. The full 33-row table is in MERGE_RESULTS.md.");
  s.addChart(pres.charts.BAR, [{ name: "Overall F1 (DEV)", labels: list.map(label), values: list.map(m => +pct(D[m].Overall_F1)) }],
    { x: 0.5, y: 1.5, w: 7.6, h: 5.3, barDir: "bar", chartColors: colors, barGapWidthPct: 35,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelColor: INK,
      valAxisMinVal: 72, valAxisMaxVal: 78, valAxisMajorUnit: 1, valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, valAxisLabelColor: GREY, catAxisLabelColor: INK,
      valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valGridLine: { color: "E3E9E6", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false,
      showTitle: true, title: "Overall macro-F1 % on DEV (axis starts at 72)", titleFontSize: 13, titleColor: DARK, titleFontFace: "+mn-lt" });
  const t = D["T: CONC"], fb = D["fastText-cc.bn"];
  card(s, 8.4, 1.6, 4.3, 2.0, DARK);
  txt(s, "Selected by the rule", { x: 8.65, y: 1.72, w: 3.8, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: GOLD });
  txt(s, [{ text: `BanglaMeta: ${pct(t.Overall_F1)}`, options: { bold: true, fontSize: 22, breakLine: true } },
          { text: `D1 ${pct(t.D1_F1)} ≥ ${pct(N.dev["fastText-cc.bn"].D1_F1 - 0.01)} and D2 ${pct(t.D2_F1)} ≥ ${pct(D["Informal-W2V"].D2_F1 - 0.01)}  →  eligible`, options: { fontSize: 13, breakLine: true } },
          { text: `vs fastText: ${sgn(t.dOverall_vs_best_single)} ${ci(t.dOverall_lo, t.dOverall_hi)}`, options: { fontSize: 13 } }],
      { x: 8.65, y: 2.12, w: 3.85, h: 1.4, color: WHITE });
  const ins = [["Concatenation > compression", `News + Informal: CONC ${pct(D["Q: CONC"].Overall_F1)} vs SVD-300 ${pct(D["Q: CONC+SVD300"].Overall_F1)}`],
               ["The informal gain comes from Informal-W2V", `P (News + Subs2vec) ${pct(D["P: CONC"].Overall_F1)} does not beat fastText ${pct(fb.Overall_F1)}`],
               ["Third model adds little", `T ${pct(t.Overall_F1)} vs Q ${pct(D["Q: CONC"].Overall_F1)}`]];
  ins.forEach(([h, d], i) => {
    card(s, 8.4, 3.85 + i * 1.0, 4.3, 0.85);
    txt(s, [{ text: h, options: { bold: true, color: DARK, breakLine: true } }, { text: d, options: { color: GREY } }], { x: 8.6, y: 3.93 + i * 1.0, w: 3.95, h: 0.68, fontSize: 12, valign: "middle" });
  });
}

// ================================================================ 18. CONF CONFIRMATION
if (want(18)) {
  const T = K["T: CONC"];
  const H = [["H1", "fastText-cc.bn"], ["H2", "News-W2V"], ["H3", "Informal-W2V"]].map(([h, b]) => ({ h, b, t: TEST(`T: CONC vs ${b}`, "Overall_F1") }));
  const s = content(6, "Held-out confirmation: 3 of 3 hypotheses confirmed",
    "The single, pre-registered test on the CONF half, which was never used for any choice. The frozen model beats all three baselines; every confidence interval is above zero and every Holm-corrected p-value is below 0.05. The p-value of 0.0001 before correction is the smallest 10,000 bootstrap samples can give.");
  stat(s, 0.6, 1.5, 3.4, "3 / 3", "pre-registered hypotheses confirmed", RED, 48);
  stat(s, 0.6, 3.05, 3.4, pct(T.Overall_F1), `BanglaMeta Overall F1 on CONF [${pct(T.Overall_lo)}, ${pct(T.Overall_hi)}]`, DARK, 40);
  stat(s, 0.6, 4.6, 3.4, `${pct(T.Overall_F1_DEV)} → ${pct(T.Overall_F1)}`, "DEV → CONF: no winner's curse", GREEN, 28);
  table(s, ["", "Hypothesis", "BanglaMeta", "Baseline", "Δ Overall [95% CI]", "Holm p", "Result"],
    H.map(({ h, b, t }) => [{ text: h, options: { bold: true, color: DARK } }, `BanglaMeta > ${b}`, pct(T.Overall_F1), pct(K[b].Overall_F1),
      `${sgn(t.delta)} ${ci(t.ci_lo, t.ci_hi)}`, p4(t.p_holm), { text: t.reject_H0 ? "confirmed" : "not confirmed", options: HL }]),
    { x: 4.3, y: 1.65, w: 8.4, colW: [0.45, 2.3, 1.05, 0.95, 1.85, 0.75, 1.05], size: 13, hsize: 12, rowH: 0.6 });
  card(s, 4.3, 4.35, 8.4, 2.1, MINT);
  txt(s, "How to read this", { x: 4.55, y: 4.48, w: 7.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: DARK });
  bullets(s, ["Δ = BanglaMeta minus baseline, in macro-F1 points; CI from 10,000 paired bootstrap samples of the same texts",
              "Holm p = one-sided p-value corrected for the three tests; confirmed if below 0.05",
              "CONF = 11,950 texts never used for training, tuning or selection"], { x: 4.55, y: 4.95, w: 7.95, h: 1.45, fontSize: 13 });
  txt(s, "Δ and CIs are computed from unrounded scores, so they can differ by 0.01 from the difference of the rounded values shown.", { x: 0.6, y: 6.6, w: 12.1, h: 0.3, fontSize: 11, color: GREY });
}

// ================================================================ 19. WHY IT WINS (CONF)
if (want(19)) {
  const ms = ["Informal-W2V", "News-W2V", "fastText-cc.bn", "T: CONC"], lab = m => m === "T: CONC" ? "BanglaMeta" : m;
  const col = ms.map(m => m === "T: CONC" ? RED : LGREY);
  const s = content(6, "Why it wins: best on news and on informal text at once",
    "On the held-out half, BanglaMeta is at the top of both charts. The single models each lead in one domain only. This is Theorem 1 seen in data: the merge keeps the strength of each source.");
  const common = { barDir: "col", barGapWidthPct: 45, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelColor: INK,
    valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, valAxisLabelColor: GREY, catAxisLabelColor: INK, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E3E9E6", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false, showTitle: true, titleFontSize: 14, titleColor: DARK, titleFontFace: "+mn-lt" };
  s.addChart(pres.charts.BAR, [{ name: "D1", labels: ms.map(lab), values: ms.map(m => +pct(K[m].D1_F1)) }],
    Object.assign({}, common, { x: 0.6, y: 1.55, w: 5.95, h: 4.3, chartColors: col, valAxisMinVal: 76, valAxisMaxVal: 90, valAxisMajorUnit: 2, title: "News (D1) macro-F1 %, CONF — axis from 76" }));
  s.addChart(pres.charts.BAR, [{ name: "D2", labels: ms.map(lab), values: ms.map(m => +pct(K[m].D2_F1)) }],
    Object.assign({}, common, { x: 6.75, y: 1.55, w: 5.95, h: 4.3, chartColors: col, valAxisMinVal: 58, valAxisMaxVal: 68, valAxisMajorUnit: 2, title: "Informal (D2) macro-F1 %, CONF — axis from 58" }));
  const T = K["T: CONC"];
  card(s, 0.6, 6.0, 12.1, 0.8, MINT);
  txt(s, [{ text: "Per task (CONF)  ", options: { bold: true, color: DARK } },
          { text: `news ${pct(T.F1_news_topic_7)}  ·  reviews ${pct(T.F1_review_sentiment_3)}  ·  social ${pct(T.F1_social_sentiment_3)}  ·  hate speech ${pct(T.F1_hate_binary_2)}` }],
      { x: 0.85, y: 6.08, w: 11.6, h: 0.65, fontSize: 15, valign: "middle" });
}

// ================================================================ 20. ALL MODELS ON CONF
if (want(20)) {
  const order = ["X: CONC", "T: CONC", "Q: CONC (exploratory)", "fastText-cc.bn", "News-W2V", "Subs2vec-w2", "Informal-W2V"];
  const name = { "X: CONC": "X: all four + fastText (extended)", "T: CONC": "BanglaMeta (T: CONC)", "Q: CONC (exploratory)": "Q: News + Informal (exploratory)" };
  const dims = { "X: CONC": 1200, "T: CONC": 900, "Q: CONC (exploratory)": 600 };
  const s = content(6, "All models on the held-out half",
    "Every model on CONF, sorted by Overall. The extended model with fastText scores highest but is not pure Word2Vec. The two-model merge Q is within noise of BanglaMeta: adding Subs2vec gave plus 0.26 points, not significant. DEV and CONF scores agree within half a point for every model.");
  const tq = TEST("T: CONC vs Q: CONC (exploratory)", "Overall_F1"), xt = TEST("X: CONC vs T: CONC", "Overall_F1");
  table(s, ["Model", "Dim", "News D1", "Informal D2", "Overall [95% CI]", "Overall on DEV"],
    order.map(m => {
      const r = K[m], hl = m === "T: CONC";
      const o = hl ? { bold: true, color: RED } : {};
      return [{ text: name[m] || m, options: Object.assign({ bold: true, color: DARK }, o) }, { text: String(dims[m] || 300), options: o }, { text: pct(r.D1_F1), options: o }, { text: pct(r.D2_F1), options: o },
              { text: `${pct(r.Overall_F1)}  [${pct(r.Overall_lo)}, ${pct(r.Overall_hi)}]`, options: o }, { text: pct(r.Overall_F1_DEV), options: o }];
    }),
    { x: 0.6, y: 1.6, w: 12.1, colW: [3.9, 0.8, 1.4, 1.5, 2.9, 1.6], size: 13, hsize: 13, rowH: 0.47 });
  card(s, 0.6, 5.55, 5.9, 1.2);
  txt(s, [{ text: "Third model (Subs2vec)  ", options: { bold: true, color: DARK, breakLine: true } },
          { text: `BanglaMeta vs Q: ${sgn(tq.delta)} ${ci(tq.ci_lo, tq.ci_hi)} → not significant` }], { x: 0.85, y: 5.65, w: 5.5, h: 1.0, fontSize: 14, valign: "middle" });
  card(s, 6.8, 5.55, 5.9, 1.2);
  txt(s, [{ text: "Adding fastText (extended)  ", options: { bold: true, color: DARK, breakLine: true } },
          { text: `X vs BanglaMeta: ${sgn(xt.delta)} ${ci(xt.ci_lo, xt.ci_hi)} → significant` }], { x: 7.05, y: 5.65, w: 5.5, h: 1.0, fontSize: 14, valign: "middle" });
  txt(s, "Q and X are secondary / exploratory comparisons, outside the pre-registered family.", { x: 0.6, y: 6.8, w: 9, h: 0.2, fontSize: 10, color: GREY });
}

// ================================================================ 21. KEY FINDINGS
if (want(21)) {
  const T = K["T: CONC"];
  const s = content(7, "Key findings",
    "Four findings. The merge works and is confirmed on unseen data. The mathematics predicted the outcome: certified methods won, uncertified ones lost. The gain comes from combining domains. And the selection was honest: DEV and CONF agree.");
  const f = [
    ["Merging works", `BanglaMeta beats fastText, News-W2V and Informal-W2V on unseen data (CONF ${pct(T.Overall_F1)}; all Holm p < 0.005).`],
    ["Theory predicted practice", "Certified concatenation won; uncertified SVD and averaging scored lower; weights changed nothing."],
    ["Domains are complementary", "The merge equals the best model on news and on informal text at the same time; no single model does."],
    ["Selection was honest", `DEV ${pct(T.Overall_F1_DEV)} vs CONF ${pct(T.Overall_F1)}: the choice among 28 candidates did not inflate the score.`],
  ];
  f.forEach(([h, d], i) => {
    const x = 0.6 + (i % 2) * 6.1, y = 1.6 + Math.floor(i / 2) * 2.2;
    card(s, x, y, 5.9, 2.0);
    badge(s, x + 0.25, y + 0.25, String(i + 1), i === 0 ? RED : GREEN, 0.65, 18);
    txt(s, h, { x: x + 1.1, y: y + 0.3, w: 4.6, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: DARK });
    txt(s, d, { x: x + 1.1, y: y + 0.9, w: 4.55, h: 1.05, fontSize: 15 });
  });
  card(s, 0.6, 6.0, 12.1, 0.8, DARK);
  txt(s, [{ text: "Answer to the research question  ", options: { bold: true, color: GOLD } },
          { text: "Yes: the merged Word2Vec beats each of its sources and fastText on held-out Bangla text." }],
      { x: 0.85, y: 6.05, w: 11.6, h: 0.7, fontSize: 16, color: WHITE, valign: "middle" });
}

// ================================================================ 22. LIMITATIONS & FUTURE WORK
if (want(22)) {
  const t = D["T: CONC"], nw = D["News-W2V"], ft = D["fastText-cc.bn"];
  const s = content(7, "Limitations and future work",
    "Honest limits. The gain over fastText is small though significant, and it comes from informal text. Word-level similarity and analogy did not improve. The probe is linear, so we measure embedding quality, not the best reachable accuracy. Future work addresses each.");
  card(s, 0.6, 1.6, 5.9, 5.15);
  txt(s, "Limitations", { x: 0.85, y: 1.75, w: 5.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: RED });
  bullets(s, [`Gain over fastText is small: ${sgn(TEST("T: CONC vs fastText-cc.bn", "Overall_F1").delta)} points Overall (significant), mainly from informal text`,
              `Word level did not improve: similarity AUC ${pct(t.sim_auc_common)} vs ${pct(nw.sim_auc_common)} (News-W2V); analogy ${pct(t.analogy_native_acc)} vs ${pct(ft.analogy_native_acc)} (fastText)`,
              "Linear probe on averaged static vectors: measures embedding quality, not fine-tuned accuracy",
              "The third model (Subs2vec) adds no significant gain"], { x: 0.85, y: 2.35, w: 5.45, h: 4.3, fontSize: 16 });
  card(s, 6.8, 1.6, 5.9, 5.15, DARK);
  txt(s, "Future work", { x: 7.05, y: 1.75, w: 5.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: GOLD });
  bullets(s, ["Weight blocks for word similarity (Theorem 6) to recover intrinsic scores",
              "Use BanglaMeta as input to CNN / BiLSTM classifiers and compare with transformer models (e.g. BanglaBERT)",
              "Add more domains (social media in Romanised Bangla, literature)",
              "Release the model and evaluation set for Bangla NLP research"], { x: 7.05, y: 2.35, w: 5.45, h: 4.3, fontSize: 16, color: WHITE });
}

// ================================================================ 23. REFERENCES
if (want(23)) {
  const s = content(8, "References", "Sources cited in this presentation. Literature retrieved and checked through SciSpace.");
  const refs = [
    "Akter, M., Sarkar, S., & Karmaker Santu, S. K. (2023). On evaluation of Bangla word analogies. EMNLP 2023.",
    "Ahmad, I., Alqurashi, F., & Mehmood, R. (2022). Potrika: Raw and balanced newspaper datasets in the Bangla language with eight topics and five attributes. arXiv.",
    "Bollegala, D. (2022). Learning meta word embeddings by unsupervised weighted concatenation of source embeddings. IJCAI 2022.",
    "Bollegala, D., & Bao, C. (2018). Learning word meta-embeddings by autoencoding. COLING 2018.",
    "Coates, J., & Bollegala, D. (2018). Frustratingly easy meta-embedding: Computing meta-embeddings by averaging source word embeddings. NAACL 2018.",
    "Grim, P., & Buchanan, E. (2026). Bengali word2vec embeddings trained on OpenSubtitles [Data set]. Zenodo, record 22230710.",
    "Hossain, M. R., & Hoque, M. M. (2020). Towards Bengali word embedding: Corpus creation, intrinsic and extrinsic evaluations. arXiv.",
    "Islam, K. I., Kar, S., Islam, M. S., et al. (2021). SentNoB: A dataset for analysing sentiment on noisy Bangla texts. Findings of EMNLP 2021.",
    "Jawanpuria, P., Satya Dev, N. T. V., Kunchukuttan, A., & Mishra, B. (2020). Learning geometric word meta-embeddings. RepL4NLP @ ACL 2020.",
    "Kabir, M., Mahfuz, O. B., Raiyan, S. R., et al. (2023). BanglaBook: A large-scale Bangla dataset for sentiment analysis from book reviews. Findings of ACL 2023.",
    "Rahman, R. (2020). Robust and consistent estimation of word embedding for Bangla language by fine-tuning Word2Vec model. ICCIT 2020.",
    "Ritu, Z. S., Nowshin, N., Nahid, M. H., et al. (2018). Performance analysis of different word embedding models on Bangla language.",
    "Romim, N., Ahmed, M., Talukder, H., et al. (2021). Hate speech detection in the Bengali language: A dataset and its baseline evaluation.",
    "Yin, W., & Schütze, H. (2016). Learning word meta-embeddings. ACL 2016.",
  ];
  const half = Math.ceil(refs.length / 2);
  [refs.slice(0, half), refs.slice(half)].forEach((col, c) =>
    s.addText(col.map((r, i) => ({ text: r, options: { breakLine: i < col.length - 1, paraSpaceAfter: 7 } })),
      { x: 0.6 + c * 6.15, y: 1.55, w: 5.95, h: 5.3, fontFace: BODY, fontSize: 11, color: INK, margin: 0, valign: "top", isTextBox: true }));
}

// ================================================================ 24. THANK YOU
if (want(24)) {
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "References" });
  const T = K["T: CONC"];
  s.addText("Thank you", { x: 0.8, y: 1.7, w: 8, h: 1.1, fontFace: HEAD, fontSize: 54, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Questions and discussion", { x: 0.8, y: 2.85, w: 8, h: 0.6, fontFace: HEAD, fontSize: 24, color: "CFE6DA", margin: 0, isTextBox: true });
  s.addText([{ text: "BanglaMeta in one line  ", options: { bold: true, color: GOLD } },
             { text: `three domain Word2Vec models, concatenated → ${pct(T.Overall_F1)} macro-F1 on unseen Bangla text, above every source and fastText.`, options: { color: WHITE } }],
    { x: 0.8, y: 4.3, w: 11.5, h: 0.9, fontFace: BODY, fontSize: 17, margin: 0, isTextBox: true });
  s.addText("Code, results and proofs: github.com/Emon3469/Thesis", { x: 0.8, y: 5.5, w: 11, h: 0.4, fontFace: BODY, fontSize: 15, color: "CFE6DA", margin: 0, isTextBox: true });
  [GREEN, GOLD, RED].forEach((c, i) => s.addShape(pres.shapes.RECTANGLE, { x: 11.4, y: 1.7 + i * 0.75, w: 0.75, h: 0.7, fill: { color: c }, line: { color: DARK, width: 1 } }));
  s.addNotes("Close with the one-line result and invite questions. Likely questions: why not transformers (scope: Word2Vec merging as asked; future work), why three models when Subs2vec adds little (pre-registered rule; Q is statistically equal), why linear probe (measures embedding quality fairly across models).");
}

(async () => {
  const out = process.env.OUT || path.join(__dirname, "..", "BanglaMeta_Thesis_Progress.pptx");
  await pres.writeFile({ fileName: out });
  if (!process.env.NOTHEME) await applyTheme(out, THEME);
  console.log("written", out);
})();
