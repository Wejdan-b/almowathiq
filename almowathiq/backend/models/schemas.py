from pydantic import BaseModel, Field
from typing import List, Optional


class Reference(BaseModel):
    """عالم أو مرجع مذكور في المصدر (من هوامش الصفحة نفسها)."""
    name: str = ""
    book: str = ""
    page: str = ""
    madhhab: str = ""
    note: str = ""


class Source(BaseModel):
    scholar: str = ""
    title: str = ""
    url: str = ""
    narrator: str = ""   # الراوي (للأحاديث)
    grade: str = ""      # درجة الحديث كما وردت في المصدر
    references: List[Reference] = Field(default_factory=list)  # للفتاوى


class VerificationResponse(BaseModel):
    content_type: str = "fatwa"

    status: str

    # سبب "يحتاج تصحيح": truncated / altered / not_authentic، وإلا None
    issue: Optional[str] = None

    # كل الأسباب مرتبة حسب الأهمية، مثل ["altered", "truncated"]
    # وقد تكون ["abridged"] مع "موثّق": نص مختصر لا يغيّر الحكم
    issues: List[str] = Field(default_factory=list)

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    extracted_text: str

    correct_text: Optional[str] = None

    # للتلوين في الواجهة
    added_words: List[str] = Field(default_factory=list)    # مضافة أو مغيّرة في الصورة (أحمر)
    removed_words: List[str] = Field(default_factory=list)  # محذوفة من المصدر (أخضر)
    source_excerpt: Optional[str] = None                    # الجملة المقابلة من المصدر

    # أدلة الفتوى كما وردت في المصدر (للعرض فقط)
    evidence: Optional[str] = None

    missing_context: List[str] = Field(
        default_factory=list
    )

    explanation: str

    source: Optional[Source] = None

    # مصدر البديل الصحيح (للأحاديث التي لا تصح فقط)
    alternative_source: Optional[Source] = None 