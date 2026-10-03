"""
البحث المحلي في قاعدة المعرفة (الجزء اللفظي من الـHybrid Retrieval).

1) توحيد النص العربي: التشكيل، والهمزات، والتاء المربوطة، والأرقام، والرموز.
2) درجة تطابق لفظي من 0 إلى 1:
   - تطابق الحروف (trigrams): يتحمّل أخطاء OCR والكلمات الملتصقة.
   - تطابق الكلمات: يكافئ وجود نفس الكلمات.
3) البحث يكون فقط في النوع المصنَّف:
   hadith  -> hadiths.json + fake_hadiths.json (المكذوب وبديله)
   fatwa   -> fatwas.json
   unknown -> لا بحث

الدرجة هنا "درجة تشابه نصي" فقط، وليست حكمًا. الحكم في مرحلة التحقق.
"""

import re
from functools import lru_cache

# ---------------------------------------------------------------
# 1) توحيد النص العربي
# ---------------------------------------------------------------
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_NON_WORD = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")

_SYMBOLS = {
    "ﷺ": " صلى الله عليه وسلم ",
    "\ufd40": " رحمه الله ",     # ﵀
    "\ufd41": " رحمه الله ",
    "\ufd42": " رحمه الله ",
    "\ufd43": " رحمه الله ",
    "\ufd47": " عليه السلام ",
    "\ufdfa": " صلى الله عليه وسلم ",
}
_LETTERS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})


def normalize_arabic(text: str) -> str:
    """يوحّد النص للمقارنة فقط. لا يُعرض للمستخدم ولا يُخزَّن."""
    if not text:
        return ""
    for sym, rep in _SYMBOLS.items():
        text = text.replace(sym, rep)
    text = _INVISIBLE.sub("", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_LETTERS)
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال")
_STOPWORDS = {
    "في", "من", "على", "عن", "الي", "ان", "او", "ثم", "قد", "لا", "ما",
    "و", "ف", "ب", "ل", "هو", "هي", "ذلك", "هذا", "هذه", "التي", "الذي",
    "كان", "قال", "قالت", "عليه", "صلي", "الله", "وسلم",
}


def tokenize(text: str) -> list:
    tokens = []
    for t in normalize_arabic(text).split():
        if t in _STOPWORDS:
            continue
        for p in _PREFIXES:
            if t.startswith(p) and len(t) - len(p) >= 2:
                t = t[len(p):]
                break
        if len(t) >= 2:
            tokens.append(t)
    return tokens


def _trigrams(text: str) -> set:
    s = normalize_arabic(text).replace(" ", "")
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s} if s else set()


# ---------------------------------------------------------------
# 2) درجة التطابق اللفظي
# ---------------------------------------------------------------
def lexical_score(query: str, doc: str) -> float:
    """
    كم من نص الاستعلام موجود في نص المصدر (0..1).
    نقيس "الاحتواء" وليس التطابق الكامل، لأن الصور كثيرًا ما تقتطع جزءًا من النص.
    """
    q_grams, d_grams = _trigrams(query), _trigrams(doc)
    q_tokens, d_tokens = set(tokenize(query)), set(tokenize(doc))
    if not q_grams:
        return 0.0
    char_score = len(q_grams & d_grams) / len(q_grams)
    token_score = len(q_tokens & d_tokens) / len(q_tokens) if q_tokens else char_score
    return round(0.7 * char_score + 0.3 * token_score, 4)


# ---------------------------------------------------------------
# 3) الفهرس والبحث
# ---------------------------------------------------------------
def build_entries(hadiths: list, fatwas: list, fake_pairs: list) -> dict:
    """يحوّل السجلات (dict) إلى مداخل قابلة للبحث، مقسّمة حسب النوع."""
    hadith_entries = [
        {"id": h["id"], "kind": "hadith", "matched": "text", "text": h["text"], "record": h}
        for h in hadiths
    ]
    for p in fake_pairs:
        hadith_entries.append({"id": p["id"], "kind": "fake_hadith", "matched": "incorrect_hadith",
                               "text": p["incorrect_hadith"]["text"], "record": p})
        hadith_entries.append({"id": p["id"], "kind": "fake_hadith", "matched": "correct_hadith",
                               "text": p["correct_hadith"]["text"], "record": p})
    fatwa_entries = [
        {"id": f["id"], "kind": "fatwa", "matched": "text",
         "text": f"{f['question']} {f['text']}", "record": f}
        for f in fatwas
    ]
    return {"hadith": hadith_entries, "fatwa": fatwa_entries}


@lru_cache(maxsize=1)
def _default_index() -> dict:
    from services.data_loader import load_all
    data = load_all(strict=True)
    return build_entries(
        [h.model_dump() for h in data["hadiths"]],
        [f.model_dump() for f in data["fatwas"]],
        [p.model_dump() for p in data["fake_hadiths"]],
    )


def search(text: str, content_type: str, top_k: int = 5, index: dict = None) -> list:
    """
    يرجع أفضل top_k مرشحين: [{id, kind, matched, score, record}, ...]
    unknown أو نص فارغ -> قائمة فارغة (لا نفرض بحثًا دينيًا).
    """
    if content_type not in ("hadith", "fatwa") or not text or not text.strip():
        return []
    entries = (index or _default_index())[content_type]

    best = {}
    for e in entries:
        score = lexical_score(text, e["text"])
        key = (e["id"], e["matched"])
        if key not in best or score > best[key]["score"]:
            best[key] = {"id": e["id"], "kind": e["kind"], "matched": e["matched"],
                         "score": score, "record": e["record"]}

    results = sorted(best.values(), key=lambda r: r["score"], reverse=True)
    return results[:top_k]
