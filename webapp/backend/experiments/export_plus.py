"""Export BanglaMeta+ as a standalone model in the standard word2vec binary format.

Input : webapp/data/models/merged_plus.f32.npy (+ words), checksum verified against data/manifest.json
Output: models/banglameta_plus/banglameta_plus_1200d.bin  (273,531 words x 1,200 dims, float32)
        models/banglameta_plus/banglameta_plus_1200d.md5
Load  : from gensim.models import KeyedVectors
        kv = KeyedVectors.load_word2vec_format("banglameta_plus_1200d.bin", binary=True)
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

man = json.loads(C.MANIFEST.read_text(encoding="utf-8"))["models"]["merged_plus"]
src = C.served_path("merged_plus")
assert md5(src) == man["served_md5"], "served BanglaMeta+ file does not match the manifest"
U = np.load(src, mmap_mode="r"); words = C.served_words("merged_plus").read_text(encoding="utf-8").split("\n")
assert U.shape == (len(words), 1200)

out_dir = C.REPO / "models" / "banglameta_plus"; out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "banglameta_plus_1200d.bin"
with open(out, "wb") as f:                                    # same layout as gensim's save_word2vec_format(binary=True)
    f.write(f"{len(words)} {U.shape[1]}\n".encode("utf-8"))
    for s in range(0, len(words), 20000):
        B = np.asarray(U[s:s + 20000], dtype="<f4")
        f.write(b"".join(f"{w} ".encode("utf-8") + row.tobytes() for w, row in zip(words[s:s + 20000], B)))
digest = md5(out)
(out_dir / "banglameta_plus_1200d.md5").write_text(f"{digest}  banglameta_plus_1200d.bin\n", encoding="utf-8")

# round trip: gensim reads the head of the file and every row equals the served row
from gensim.models import KeyedVectors
kv = KeyedVectors.load_word2vec_format(str(out), binary=True, limit=5000)
assert kv.index_to_key == words[:5000] and np.array_equal(kv.vectors, np.asarray(U[:5000]))
print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GiB), md5 {digest}; gensim round trip on 5,000 rows exact")
