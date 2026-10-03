"""
تحميل قاعدة المعرفة المحلية والتحقق من سلامتها.

الملفات:
  data/hadiths.json       أحاديث موثقة المصدر
  data/fatwas.json        فتاوى من مصادر موثوقة
  data/fake_hadiths.json  أحاديث منتشرة لا تصح + بديلها الصحيح في نفس السجل

القاعدة: أي سجل ناقص المصدر أو المرجع يُرفض ولا يدخل النظام.
"""

import json
import re
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# حروف مخفية تكسر المطابقة أو ملفات JSON
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")


def _check_clean(value: str) -> str:
    if _INVISIBLE.search(value):
        raise ValueError("يحتوي على حرف مخفي")
    if _MARKDOWN_LINK.search(value):
        raise ValueError("يحتوي على رابط بصيغة Markdown")
    return value


class _Base(BaseModel):
    model_config = {"extra": "forbid"}

    @field_validator("*", mode="before")
    @classmethod
    def _strings_clean(cls, v):
        if isinstance(v, str):
            return _check_clean(v)
        return v

    @field_validator("reference_url", check_fields=False)
    @classmethod
    def _url_ok(cls, v):
        if v is not None and not v.startswith("https://"):
            raise ValueError("الرابط يجب أن يبدأ بـ https://")
        return v


class Hadith(_Base):
    id: str = Field(pattern=r"^H\d{4}$")
    type: Literal["hadith"]
    text: str = Field(min_length=5)
    narrator: str = Field(min_length=2)
    source: str = Field(min_length=2)
    number: str = Field(min_length=1)
    grade: str = Field(min_length=2)
    grader: str = Field(min_length=2)
    grade_source: str = Field(min_length=2)
    also_in: List[str] = []
    reference_url: Optional[str] = None


class ScholarRef(_Base):
    """عالم أو مرجع مذكور في هوامش صفحة المصدر نفسها."""
    name: str = ""          # اسم العالم كما ورد في الصفحة
    madhhab: str = ""       # المذهب إن ذكرته الصفحة
    book: str = Field(min_length=2)
    page: str = ""          # المجلد/الصفحة كما وردت
    note: str = ""          # مثل: حكاية الإجماع


class Fatwa(_Base):
    id: str = Field(pattern=r"^F\d{4}$")
    type: Literal["fatwa"]
    question: str = Field(min_length=5)
    text: str = Field(min_length=5)
    mufti: str = ""  # قد يكون فارغًا إذا كان الحكم من موسوعة وليس من مفتٍ بعينه
    source: str = Field(min_length=2)
    number: Optional[str] = None
    topic: str = Field(min_length=2)
    madhhab: str = Field(min_length=2)
    is_khilafi: bool
    reference_url: Optional[str] = None
    scholars: List[ScholarRef] = []


class FakeEntry(_Base):
    """الحديث المنتشر الذي حكم عليه المصدر بأنه لا يصح."""
    text: str = Field(min_length=3)
    narrator: str = ""
    source: str = ""
    number: str = ""
    grade: str = Field(min_length=2)
    grader: str = ""
    grade_source: str = Field(min_length=2)
    reference_url: str  # إلزامي: حكم "لا يصح" يجب أن يكون له مرجع


class CorrectEntry(_Base):
    """البديل الصحيح الذي يُعرض للمستخدم بدل الحديث المنتشر."""
    text: str = Field(min_length=5)
    narrator: str = Field(min_length=2)
    source: str = Field(min_length=2)
    number: str = Field(min_length=1)
    grade: str = Field(min_length=2)
    grader: str = Field(min_length=2)
    grade_source: str = Field(min_length=2)
    also_in: List[str] = []
    reference_url: Optional[str] = None


class FakeHadith(_Base):
    """زوج: حديث منتشر لا يصح + بديله الصحيح، في سجل واحد للوصول السريع."""
    id: str = Field(pattern=r"^HF\d{4}$")
    type: Literal["hadith"]
    incorrect_hadith: FakeEntry
    correct_hadith: CorrectEntry

    @property
    def text(self) -> str:
        return self.incorrect_hadith.text


FILES = {
    "hadiths.json": Hadith,
    "fatwas.json": Fatwa,
    "fake_hadiths.json": FakeHadith,
}


class DataError(Exception):
    pass


def _load_file(filename: str, model) -> tuple[list, list]:
    path = DATA_DIR / filename
    errors = []
    if not path.exists():
        return [], [f"{filename}: الملف غير موجود"]

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return [], [f"{filename}: الملف فارغ"]

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], [f"{filename}: JSON غير صالح (سطر {e.lineno}): {e.msg}"]

    if not isinstance(items, list):
        return [], [f"{filename}: يجب أن يكون قائمة [ ... ]"]

    records = []
    for i, item in enumerate(items, 1):
        rid = item.get("id", f"#{i}") if isinstance(item, dict) else f"#{i}"
        try:
            records.append(model.model_validate(item))
        except ValidationError as e:
            for err in e.errors():
                field = ".".join(str(p) for p in err["loc"]) or "-"
                errors.append(f"{filename} | {rid} | {field}: {err['msg']}")
    return records, errors


def load_all(strict: bool = True) -> dict:
    """
    يحمّل كل الملفات ويتحقق منها.
    strict=True: يرفع DataError عند أي خطأ (للاستخدام داخل السيرفر).
    """
    data, errors = {}, []
    for filename, model in FILES.items():
        records, errs = _load_file(filename, model)
        data[filename] = records
        errors.extend(errs)

    # تكرار الـid في كل الملفات
    seen = {}
    for filename, records in data.items():
        for r in records:
            if r.id in seen:
                errors.append(f"id مكرر: {r.id} في {seen[r.id]} و {filename}")
            seen[r.id] = filename

    # تكرار النص داخل نفس الملف
    for filename, records in data.items():
        texts = {}
        for r in records:
            if r.text in texts:
                errors.append(f"{filename}: نص مكرر في {texts[r.text]} و {r.id}")
            texts[r.text] = r.id

    if strict and errors:
        raise DataError("\n".join(errors))

    return {
        "hadiths": data["hadiths.json"],
        "fatwas": data["fatwas.json"],
        "fake_hadiths": data["fake_hadiths.json"],
        "errors": errors,
    }
