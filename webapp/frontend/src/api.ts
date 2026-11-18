// Typed client for the BanglaMeta API. Every value shown in the UI comes from these responses.

export type ModelKey = "merged_plus" | "merged" | "merged_sim" | "fasttext" | "news" | "informal";
export type View = "merged_plus" | "merged" | "merged_sim";
export type Projection = "pca" | "pca-local" | "tsne";

export interface Resolved {
  input: string;
  kind: "bangla" | "banglish" | "number" | "other";
  word: string | null;
  options: string[];
  found: boolean;
  in_model?: boolean;
}
export interface Neighbor { word: string; cos: number; rank: number }
export interface Provenance { news: boolean; informal: boolean; subs: boolean }
export interface WordResult extends Resolved {
  in_model: boolean;
  row: number | null;
  provenance: Provenance | null;
  neighbors: Neighbor[];
}
export interface SentenceResult {
  text: string;
  tokens: (Resolved & { in_model: boolean })[];
  used: number;
  coverage: number;
  nearest: Neighbor[];
}
export interface Point {
  id: string;
  label: string;
  kind: "query" | "neighbor" | "sentence" | "token";
  groups: string[];
  row?: number;
  sentence?: number;
  xyz: [number, number, number];
}
export interface Edge { source: string; target: string; cos: number }
export interface Relation {
  source: string; target: string; cos: number; angle_deg: number;
  rank_ab: number | null; rank_ba: number | null; strength: string | null;
  bridges: { word: string; cos_a: number; cos_b: number }[];
}
export interface ExploreResponse {
  relations: Relation[];
  model: ModelKey;
  projection: { method: Projection; dims: number; explained: number[] | null };
  words: WordResult[];
  sentences: SentenceResult[];
  sentence_similarity: (number | null)[][];
  points: Point[];
  edges: Edge[];
}
export interface ModelInfo {
  key: ModelKey; name: string; label: string; color: string; about: string; rows: number; dim: number;
  source_md5: string; served_md5: string; analogy_search_rows: number; pca_explained: number[];
}
export interface ModelsResponse {
  order: ModelKey[]; primary: ModelKey; frozen_md5: string;
  checks: { union_vocab: boolean; news_block_min_cos: number; subs_rows: number; provenance_counts: Record<string, number> };
  models: Record<ModelKey, ModelInfo>;
}
export interface CloudResponse { model: ModelKey; words: string[]; points: [number, number, number][]; explained: number[] }
export interface AnalogyResponse {
  model: ModelKey; a: Resolved; b: Resolved; c: Resolved; rule: string; ok: boolean; missing: string[];
  results: Neighbor[]; points?: number[][];
}
export interface CompareWordResponse {
  query: Resolved; k: number;
  models: Record<ModelKey, { in_model: boolean; row: number | null; neighbors: Neighbor[]; provenance: Provenance | null }>;
  overlap: Record<string, number | null>;
}
export interface CompareSentResponse {
  sentences: { text: string; tokens: Resolved[] }[];
  models: Record<ModelKey, { similarity: (number | null)[][]; coverage: number[]; nearest: Neighbor[][] }>;
}
export interface SimilarityResponse { results: { a: Resolved; b: Resolved; cos: Record<ModelKey, number | null> }[] }
export type Metrics = Record<string, number | null | undefined> & { analogy_per_file?: unknown };
export interface BenchResponse {
  measured: null | { created: string; seconds: number; results: Record<ModelKey, Metrics>; thesis: Record<ModelKey, Metrics> };
  job: { running: boolean; message: string; progress: number; error: string | null };
  official_conf: {
    models: Record<ModelKey, Record<string, number>>;
    tests: { comparison: string; delta: number; ci_lo: number; ci_hi: number; p_holm: number; reject_H0: boolean }[];
    source: string;
  };
  thesis_dev: Record<ModelKey, Metrics>;
}

export interface PairItem { kind: "word" | "sentence"; text: string; id: string | null; label: string }
export type PairSide =
  | (Resolved & { kind_of_item: "word"; in_model: boolean })
  | { kind_of_item: "sentence"; text: string; tokens: (Resolved & { in_model: boolean })[]; used: number; in_model: boolean };
export interface PairResponse { model: ModelKey; a: PairSide; b: PairSide; cos: number | null; angle_deg: number | null }

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("The model server is not reachable. Is the backend running on port 8000?", 0);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    let msg: unknown = data?.detail;
    if (Array.isArray(msg)) msg = msg.map((d: { msg?: string; loc?: unknown[] }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`).join("; ");
    throw new ApiError(typeof msg === "string" && msg ? msg : `Server error ${res.status}`, res.status);
  }
  return data as T;
}

export const api = {
  models: () => request<ModelsResponse>("/models"),
  resolve: (text: string, signal?: AbortSignal) =>
    request<{ text: string; tokens: Resolved[] }>(`/resolve?text=${encodeURIComponent(text)}`, undefined, signal),
  explore: (b: { words: string[]; sentences: string[]; k: number; projection: Projection; dims: 2 | 3; similar: boolean; model: View }, signal?: AbortSignal) =>
    request<ExploreResponse>("/explore", b, signal),
  cloud: (model: ModelKey = "merged", n = 2500) => request<CloudResponse>(`/cloud?model=${model}&n=${n}`),
  analogy: (a: string, b: string, c: string, k = 10, model: View = "merged_plus") => request<AnalogyResponse>("/analogy", { a, b, c, k, model }),
  compareWord: (word: string, k: number) => request<CompareWordResponse>("/compare/word", { word, k }),
  compareSentences: (sentences: string[]) => request<CompareSentResponse>("/compare/sentences", { sentences }),
  similarity: (pairs: [string, string][]) => request<SimilarityResponse>("/similarity", { pairs }),
  pair: (a: { kind: "word" | "sentence"; text: string }, b: { kind: "word" | "sentence"; text: string }, model: View = "merged_plus") =>
    request<PairResponse>("/pair", { a, b, model }),
  benchmarks: () => request<BenchResponse>("/benchmarks"),
  runBenchmarks: () => request<{ started: boolean }>("/benchmarks/run", {}),
};

// Same character rule as the server (app/text.py): Bangla script, Latin letters, digits, punctuation, spaces.
const BENGALI = /[ঀ-৿]/;
const LATIN = /[A-Za-z]/;
export function checkBanglaInput(s: string): string | null {
  for (const ch of s) {
    if (BENGALI.test(ch) || LATIN.test(ch) || /\s/.test(ch) || /[0-9]/.test(ch) || /\p{P}/u.test(ch) || /[​-‍﻿]/.test(ch)) continue;
    return `Only Bangla (বাংলা) or Banglish (Latin letters) are accepted — "${ch}" is not.`;
  }
  return null;
}
