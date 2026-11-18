"""Tokenizer parity with the thesis, input rule, and Banglish transliteration."""
import pandas as pd
import pytest

import config as C
from app import text as T


def test_tokenizer_is_the_thesis_tokenizer():
    ev = pd.read_parquet(C.EVAL_DATA / "eval_v2.parquet", columns=["text", "tokens"]).sample(3000, random_state=0)
    for text, toks in zip(ev.text, ev.tokens):
        assert T.tokenize(text) == toks.split()


@pytest.mark.parametrize("s", ["আমি ভালো আছি", "ami bhalo achi", "আমি bhalo আছি!", "দাম ১০০ টাকা।", "kemon acho?", "ক্ষমা—জ্ঞান"])
def test_accepts_bangla_and_banglish(s):
    assert T.check_text(s, 600, "x")


@pytest.mark.parametrize("s", ["नमस्ते", "привет", "你好", "مرحبا", "café", "hello 😀", "", "   ", "123 456", "!!!"])
def test_rejects_other_scripts_and_empty(s):
    with pytest.raises(T.InputError):
        T.check_text(s, 600, "x")


def test_rejects_too_long():
    with pytest.raises(T.InputError):
        T.check_text("ক" * 601, 600, "x")


EXPECTED = {
    "ami": "আমি", "tumi": "তুমি", "kemon": "কেমন", "bangladesh": "বাংলাদেশ", "dhaka": "ঢাকা", "khela": "খেলা",
    "sundor": "সুন্দর", "shundor": "সুন্দর", "jonno": "জন্য", "onno": "অন্য", "kintu": "কিন্তু", "amra": "আমরা",
    "bhai": "ভাই", "boi": "বই", "ar": "আর", "na": "না", "khub": "খুব", "kharap": "খারাপ", "desh": "দেশ",
    "manush": "মানুষ", "taka": "টাকা", "sobai": "সবাই", "kotha": "কথা", "onek": "অনেক", "ekta": "একটা",
    "shikkha": "শিক্ষা", "khoma": "ক্ষমা", "gyan": "জ্ঞান", "hoy": "হয়", "bangla": "বাংলা", "ache": "আছে",
    "rani": "রানী", "nari": "নারী", "purush": "পুরুষ", "raja": "রাজা", "mohila": "মহিলা", "chele": "ছেলে",
}


@pytest.mark.parametrize("latin,bangla", EXPECTED.items())
def test_banglish_top_candidate(R, latin, bangla):
    import unicodedata
    c = R.translit.candidates(latin)
    assert c and c[0][0] == unicodedata.normalize("NFC", bangla)


@pytest.mark.parametrize("latin,bangla", [("valo", "ভালো"), ("bhalo", "ভালো"), ("bristi", "বৃষ্টি"), ("chand", "চাঁদ"), ("acho", "আছো"), ("korbo", "করবো")])
def test_banglish_candidate_listed(R, latin, bangla):
    assert bangla in [w for w, _, _ in R.translit.candidates(latin)]


def test_every_candidate_is_a_model_word(R):
    for latin in list(EXPECTED) + ["bhalobasha", "bolechilo", "porikkha", "xyzqq"]:
        for w, rank, _ in R.translit.candidates(latin, limit=20):
            assert R.merged.has(w) and R.merged.idx[w] == rank


@pytest.mark.parametrize("bad", ["", "qqqqzzzz", "a1b", "x" * 31])
def test_no_candidates_for_nonsense(R, bad):
    assert R.translit.candidates(bad) == []
