from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import VerificationResponse
from services.ocr import extract_text
from services.classifier import classify_content
from services.search import search
from services.verifier import verify


def warm_up():
    """
    يجهز قاعدة المعرفة وفهرس البحث بالمعنى عند تشغيل السيرفر،
    حتى لا ينتظر أول مستخدم بناء المتجهات.
    لا يمنع تشغيل السيرفر إذا فشل (يُبنى لاحقًا عند أول طلب).
    """
    try:
        from services.search import _default_index, _semantic_index
        index = _default_index()
        for content_type in ("hadith", "fatwa"):
            _semantic_index(content_type, index[content_type])
        print(
            "WARM-UP: ready "
            f"({len(index['hadith'])} hadith entries, {len(index['fatwa'])} fatwa entries)"
        )
    except Exception as error:
        print(f"WARM-UP skipped: {type(error).__name__}: {error}")


@asynccontextmanager
async def lifespan(app):
    warm_up()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="الموثّق الذكي API",
    description="Backend for Al-Mowathiq hackathon MVP",
    version="0.1.0",
)


# السماح لتطبيق Flutter Web بالاتصال بالـ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _text(value) -> str:
    return "" if value is None else str(value)


def build_source(candidate: dict) -> dict:
    """يبني بيانات المصدر للعرض من السجل المطابق (بدون أي إضافة من خارجه)."""
    record = candidate.get("record") or {}
    kind = candidate.get("kind")
    matched = candidate.get("matched")

    if kind == "fake_hadith":
        if matched == "incorrect_hadith":
            data = record.get("incorrect_hadith") or {}
            # حكم المصدر على الحديث المنتشر
            return {
                "scholar": _text(data.get("grade_source")),
                "title": f"الحكم: {_text(data.get('grade'))}",
                "url": _text(data.get("reference_url")),
            }
        data = record.get("correct_hadith") or {}
        title = _text(data.get("source"))
        if data.get("number"):
            title += f" ({data.get('number')})"
        return {
            "scholar": _text(data.get("grader")),
            "title": title,
            "url": _text(data.get("reference_url")),
            "narrator": _text(data.get("narrator")),
            "grade": _text(data.get("grade")),
        }

    if kind == "hadith":
        title = _text(record.get("source"))
        if record.get("number"):
            title += f" ({record.get('number')})"
        return {
            "scholar": _text(record.get("grader")),
            "title": title,
            "url": _text(record.get("reference_url")),
            "narrator": _text(record.get("narrator")),
            "grade": _text(record.get("grade")),
        }

    if kind == "fatwa":
        references = []
        for item in record.get("scholars") or []:
            if isinstance(item, dict):
                references.append({
                    "name": _text(item.get("name")),
                    "book": _text(item.get("book")),
                    "page": _text(item.get("page")),
                    "madhhab": _text(item.get("madhhab")),
                    "note": _text(item.get("note")),
                })
        return {
            "scholar": _text(record.get("mufti")),
            "title": _text(record.get("source")),
            "url": _text(record.get("reference_url")),
            "references": references,
        }

    return None



@app.get("/")
def root():
    return {
        "project": "الموثّق الذكي",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post(
    "/verify",
    response_model=VerificationResponse,
)
async def verify_image(
    image: UploadFile = File(...),
):
    if not image.content_type:
        raise HTTPException(
            status_code=400,
            detail="نوع الملف غير معروف",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="يسمح فقط بصور JPG أو PNG أو WEBP",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="الصورة فارغة",
        )

    try:
        # 1. استخراج النص من الصورة
        extracted = extract_text(
            image_bytes,
            image.content_type,
        )

        extracted_text = extracted.get("text", "")

        if not extracted_text:
            return VerificationResponse(
                content_type="unknown",
                status="غير موثّق",
                confidence=0.0,
                extracted_text="",
                correct_text=None,
                missing_context=[],
                explanation=(
                    "لم نتمكن من استخراج نص واضح من الصورة."
                ),
                source=None,
            )

        # 2. تصنيف المحتوى
        classification = classify_content(
            extracted_text
        )

        if classification.get("error"):
            # تعذر الاتصال بخدمة التصنيف: لا نقول للمستخدم إن المحتوى غير ديني
            return VerificationResponse(
                content_type="unknown",
                status="غير موثّق",
                confidence=0.0,
                extracted_text=extracted_text,
                correct_text=None,
                missing_context=[],
                explanation=(
                    "تعذر إجراء التحقق حاليًا بسبب ضغط مؤقت "
                    "في خدمة التحقق. يرجى المحاولة مرة أخرى."
                ),
                source=None,
            )

        content_type = classification.get(
            "content_type",
            "unknown",
        )

        if content_type not in {"hadith", "fatwa"}:
            # شبكة أمان: قبل الاستسلام، نبحث في النوعين.
            # إذا وُجد سجل مطابق بقوة، نكمل التحقق بنوعه حتى لا يضيع نص موجود عندنا.
            best = None
            for ctype in ("hadith", "fatwa"):
                found = search(extracted_text, ctype, top_k=1)
                if found and (best is None or found[0]["score"] > best[1]["score"]):
                    best = (ctype, found[0])
            if best and (
                best[1].get("lexical_score", 0) >= 0.85 or best[1]["score"] >= 0.6
            ):
                content_type = best[0]

        if content_type not in {"hadith", "fatwa"}:
            return VerificationResponse(
                content_type="unknown",
                status="غير موثّق",
                confidence=0.0,
                extracted_text=extracted_text,
                correct_text=None,
                missing_context=[],
                explanation=(
                    "لم نتمكن من تحديد نوع المحتوى "
                    "للتحقق منه."
                ),
                source=None,
            )

        # 3. البحث في قاعدة المعرفة
        candidates = search(
            extracted_text,
            content_type,
            top_k=5,
        )

        # 4. التحقق من النص
        result = verify(
            extracted_text,
            content_type,
            candidates,
        )

        # 5. تحديد المصدر (فقط إذا وُجد تطابق؛ "غير موثّق" بدون مصدر دائمًا)
        status = result.get("status", "غير موثّق")
        matched_id = result.get("matched_id")
        matched_candidate = None

        if status != "غير موثّق":
            for candidate in candidates:
                if candidate.get("id") == matched_id:
                    matched_candidate = candidate
                    break

        source = build_source(matched_candidate) if matched_candidate else None

        # أدلة الفتوى (تُعرض فقط عند وجود تطابق)
        evidence = None
        if matched_candidate and matched_candidate.get("kind") == "fatwa":
            evidence = (matched_candidate.get("record") or {}).get("evidence") or None

        # للحديث الذي لا يصح: نرسل أيضًا مصدر البديل الصحيح
        alternative_source = None
        if (
            matched_candidate
            and matched_candidate.get("kind") == "fake_hadith"
            and result.get("issue") == "not_authentic"
        ):
            alternative_source = build_source(
                {**matched_candidate, "matched": "correct_hadith"}
            )

        return VerificationResponse(
            content_type=content_type,
            status=status,
            issue=result.get("issue"),
            issues=result.get("issues") or [],
            added_words=result.get("added_words") or [],
            removed_words=result.get("removed_words") or [],
            source_excerpt=result.get("source_excerpt"),
            confidence=float(
                result.get(
                    "confidence",
                    0.0,
                )
            ),
            extracted_text=extracted_text,
            correct_text=result.get(
                "correct_text"
            ),
            missing_context=[],
            explanation=result.get(
                "explanation",
                "",
            ),
            source=source,
            alternative_source=alternative_source,
            evidence=evidence,
        )

    except Exception as error:
        print(
            f"VERIFY ERROR: {type(error).__name__}: {error}"
        )

        return VerificationResponse(
            content_type="unknown",
            status="غير موثّق",
            confidence=0.0,
            extracted_text="",
            correct_text=None,
            missing_context=[],
            explanation=(
                "تعذر إجراء التحقق حاليًا بسبب ضغط أو "
                "تعذر مؤقت في خدمة التحقق. "
                "يرجى المحاولة مرة أخرى."
            ),
            source=None,
        ) 