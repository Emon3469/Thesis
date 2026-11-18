"""Banglish (Bangla written in Latin letters) -> Bangla script, constrained to the merged model's vocabulary.

How it works (deterministic, no learned component):
  * A phonetic table lists the Bangla spellings each Latin letter group can stand for (e.g. "sh" -> শ / ষ / স).
  * A beam search reads the Latin word left to right and builds Bangla strings. After every step a string is kept
    only if some word of the merged model starts with it, and at the end only complete model words survive.
  * Candidates are ranked by the word's frequency rank in the model (its row order), then by how many
    non-default spellings were used.
So every candidate is a real row of the merged model; when none exists the word is reported as not found, never
replaced by a made-up vector.
"""
import math
from bisect import bisect_left
from dataclasses import dataclass

HASANTA = "্"
YA_PHALA = HASANTA + "য"

# (independent form, dependent sign); "" = inherent vowel (nothing written)
VOWELS = {
    "aa": [("আ", "া")],
    "a": [("আ", "া"), ("অ", ""), ("এ", "ে")],
    "ee": [("ঈ", "ী"), ("ই", "ি")],
    "ii": [("ঈ", "ী"), ("ই", "ি")],
    "i": [("ই", "ি"), ("ঈ", "ী")],
    "oo": [("উ", "ু"), ("ঊ", "ূ")],
    "uu": [("ঊ", "ূ"), ("উ", "ু")],
    "u": [("উ", "ু"), ("ঊ", "ূ")],
    "e": [("এ", "ে"), ("অ্যা", "্যা"), ("ই", "ি")],
    "oi": [("ঐ", "ৈ"), ("ওই", "োই")],
    "ou": [("ঔ", "ৌ"), ("ওউ", "োউ")],
    "o": [("অ", ""), ("ও", "ো")],
    "w": [("ও", "ো")],
    "ri": [("ঋ", "ৃ")],          # also read as r + i through the consonant table
}
RA, RHA, YA_ANTA = "ড়", "ঢ়", "য়"     # ড় ঢ় য় (decomposed, as NFC keeps them)
CONSONANTS = {
    "kkh": ["ক্ষ"], "kh": ["খ", "ক্ষ"], "k": ["ক"], "q": ["ক"], "x": ["ক্স"],
    "gh": ["ঘ"], "gg": ["জ্ঞ"], "gy": ["জ্ঞ"], "g": ["গ"],
    "chh": ["ছ"], "ch": ["চ", "ছ"], "c": ["চ", "ছ"],
    "jh": ["ঝ"], "j": ["জ", "য"], "z": ["জ", "য"],
    "th": ["থ", "ঠ"], "t": ["ত", "ট"],
    "dh": ["ধ", "ঢ"], "d": ["দ", "ড"],
    "n": ["ন", "ণ"],
    "ph": ["ফ"], "f": ["ফ"], "p": ["প"],
    "bh": ["ভ"], "v": ["ভ"], "b": ["ব"],
    "m": ["ম"],
    "rh": [RHA, RA], "r": ["র", RA],
    "l": ["ল"],
    "sh": ["শ", "ষ", "স"], "s": ["স", "শ", "ষ"],
    "h": ["হ"],
    "y": [YA_ANTA, "য"],
}
# signs that close a syllable: ং ঃ ঁ ৎ ঙ
SIGNS = {"ng": ["ং", "ঙ", "ঙ্গ"], "n": ["ং", "ঁ"], "h": ["ঃ"], "t": ["ৎ"]}
KEYS = sorted(set(VOWELS) | set(CONSONANTS) | set(SIGNS), key=len, reverse=True)
BEAM = 4000
COST_WEIGHT = 1.5
EITHER_LENGTH = {"i", "ee", "ii", "u", "oo", "uu"}


@dataclass(frozen=True)
class State:
    out: str          # Bangla built so far
    prev: str         # "S" start, "C" consonant, "V" vowel, "X" sign
    prev_roman: str   # Latin group of the previous consonant (for doubled letters -> ya-phala)
    cost: int         # number of non-default choices


class Transliterator:
    def __init__(self, words_in_rank_order):
        self.rank = {w: i for i, w in enumerate(words_in_rank_order)}
        self.sorted = sorted(self.rank)

    def _has_prefix(self, p: str) -> bool:
        i = bisect_left(self.sorted, p)
        return i < len(self.sorted) and self.sorted[i].startswith(p)

    def _steps(self, st: State, key: str):
        """All (new_out, prev, prev_roman, extra_cost) for reading Latin group `key` in state `st`."""
        if key in VOWELS:
            for j, (ind, dep) in enumerate(VOWELS[key]):
                # Banglish writes short and long i / u the same way (rani = রানী or রানি): no penalty, frequency decides
                yield (st.out + (dep if st.prev == "C" else ind)), "V", "", (0 if key in EITHER_LENGTH else j)
        if key in CONSONANTS:
            for j, b in enumerate(CONSONANTS[key]):
                if st.prev == "C":
                    yield st.out + HASANTA + b, "C", key, j            # conjunct (no vowel between)
                    yield st.out + b, "C", key, j + 1                  # inherent vowel between
                else:
                    yield st.out + b, "C", key, j
            if st.prev == "C" and key == st.prev_roman:               # doubled letter often written for ya-phala: jonno -> জন্য
                yield st.out + YA_PHALA, "C", "", 1
            if key == "y" and st.prev == "C":
                yield st.out + YA_PHALA, "C", "", 0
        if key in SIGNS and st.prev in ("V", "C"):
            for j, b in enumerate(SIGNS[key]):
                yield st.out + b, "X", "", j + 1

    def candidates(self, latin: str, limit: int = 8):
        """Model words the Latin spelling can stand for, best first: [(word, rank, cost)]."""
        s = latin.lower()
        if not s or len(s) > 30 or not s.isascii() or not s.isalpha():
            return []
        beams = {0: {State("", "S", "", 0)}}
        done = {}
        for i in range(len(s) + 1):
            states = beams.pop(i, set())
            if i == len(s):
                for st in states:
                    if st.out in self.rank and st.cost < done.get(st.out, 1 << 30):
                        done[st.out] = st.cost
                break
            if len(states) > BEAM:                                     # keep the cheapest states
                states = set(sorted(states, key=lambda x: (x.cost, x.out))[:BEAM])
            for st in states:
                for key in KEYS:
                    if not s.startswith(key, i):
                        continue
                    for out, prev, pr, extra in self._steps(st, key):
                        if self._has_prefix(out):
                            beams.setdefault(i + len(key), set()).add(State(out, prev, pr, st.cost + extra))
        # frequency (log row rank) plus a fixed penalty per non-default spelling choice
        ranked = sorted(done.items(), key=lambda kv: (math.log(self.rank[kv[0]] + 2) + COST_WEIGHT * kv[1], self.rank[kv[0]]))
        return [(w, self.rank[w], c) for w, c in ranked[:limit]]
