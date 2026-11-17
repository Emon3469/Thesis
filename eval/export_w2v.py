"""Export the frozen merged model (Step 4 selection "T: CONC") to the standard word2vec binary format.

Input : merge_output/merge_results/models/selected_pure_w2v.f16.npy (+ .words.txt), MD5 10aa206f8462c82a7b16cf60054be70f
Output: ../models/merged_w2v_T_conc/bangla_merged_w2v_900d.bin (273,531 words x 900 dims)
Usage : python export_w2v.py
"""
import hashlib
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors

HERE = Path(__file__).parent
SRC = HERE / "merge_output" / "merge_results" / "models"
OUT = HERE.parent / "models" / "merged_w2v_T_conc" / "bangla_merged_w2v_900d.bin"
FROZEN_MD5 = "10aa206f8462c82a7b16cf60054be70f"

f = SRC / "selected_pure_w2v.f16.npy"
h = hashlib.md5()
with open(f, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
assert h.hexdigest() == FROZEN_MD5, "the model file is not the frozen Step 4 selection"

U = np.load(f).astype(np.float32)
words = (SRC / "selected_pure_w2v.words.txt").read_text(encoding="utf-8").split("\n")
assert len(words) == U.shape[0] and len(set(words)) == len(words)

kv = KeyedVectors(vector_size=U.shape[1]); kv.add_vectors(words, U)
OUT.parent.mkdir(parents=True, exist_ok=True); kv.save_word2vec_format(str(OUT), binary=True)

back = KeyedVectors.load_word2vec_format(str(OUT), binary=True)
assert len(back) == len(words) and np.abs(back[words[0]] - U[0]).max() == 0
print(f"saved {OUT} ({len(back):,} words x {back.vector_size} dims); round trip exact")
