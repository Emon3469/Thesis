# Merged Bangla Word2Vec (News-W2V + Informal-W2V + Subs2vec-w2), 900-d

- File: `bangla_merged_w2v_900d.bin` (standard word2vec binary format, 273,531 words × 900 dimensions).
- Load: `from gensim.models import KeyedVectors; kv = KeyedVectors.load_word2vec_format("bangla_merged_w2v_900d.bin", binary=True)`
- Built by Step 4 (`eval/merge_src.py`), selected on the DEV half by the pre-registered rule ("T: CONC").
- Frozen source: `eval/merge_output/merge_results/models/selected_pure_w2v.f16.npy` (MD5 10aa206f8462c82a7b16cf60054be70f), values copied without change.

## How each row is made
m(w) = (1/√3) · [ news(w) ; informal(w) ; subs(w) ]

- Each block is the L2-normalised 300-d row from its source model.
- A word missing from a source gets that block predicted from another source by a ridge map fitted on the shared words, then normalised.
- Every row has length 1, and each source contributes one third of it.
- Columns 1–300 come from News-W2V, 301–600 from Informal-W2V and 601–900 from Subs2vec-w2.

## Use
- Built for text classification over averaged word vectors. On the held-out CONF half its Overall macro-F1 is 0.7717, above fastText (0.7644), News-W2V (0.7477) and Informal-W2V (0.7326); all three gains are significant after Holm correction (`eval/CONF_RESULTS.md`).
- Not the best choice for word similarity or analogy: News-W2V and fastText score higher there (see `eval/MERGE_RESULTS.md`).
