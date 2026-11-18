"""All-but-the-top (Mu & Viswanath, ICLR 2018) applied to each 300-d block of the frozen merged rows.

For every block: subtract the block mean and the projections on its top D principal directions (both fitted on the
50,000 most frequent rows), then re-normalise the block to length 1/sqrt(3). Writes data/models/merged_abtt<D>.f32.npy.
Usage: python build_abtt.py D
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

D = int(sys.argv[1])
U = np.load(C.served_path("merged"), mmap_mode="r")
out = np.lib.format.open_memmap(C.DERIVED / f"merged_abtt{D}.f32.npy", mode="w+", dtype=np.float32, shape=U.shape)
params = []
for k in range(3):
    bk = slice(300 * k, 300 * (k + 1))
    X = np.asarray(U[:50_000, bk], np.float64)
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False); P = Vt[:D]
    params.append((bk, mu.astype(np.float32), P.astype(np.float32)))
for s in range(0, U.shape[0], 16384):
    B = np.asarray(U[s:s + 16384], np.float32); o = np.empty_like(B)
    for bk, mu, P in params:
        Y = B[:, bk] - mu; Y -= (Y @ P.T) @ P
        o[:, bk] = Y / np.linalg.norm(Y, axis=1, keepdims=True) / np.sqrt(3)
    out[s:s + 16384] = o
out.flush()
np.savez(C.DERIVED / f"merged_abtt{D}.params.npz", **{f"mu{k}": p[1] for k, p in enumerate(params)}, **{f"P{k}": p[2] for k, p in enumerate(params)})
print("wrote", D)
