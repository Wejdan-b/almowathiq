from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import VerificationResponse
from services.ocr import extract_text
from services.classifier import classify_content
from services.search import search
from services.verifier import verify


app = FastAPI(
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

        content_type = classification.get(
            "content_type",
            "unknown",
        )

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

        # 5. تحديد المصدر
        matched_id = result.get("matched_id")
        matched_candidate = None

        for candidate in candidates:
            if candidate.get("id") == matched_id:
                matched_candidate = candidate
                break

        source = None

        if matched_candidate:
            record = matched_candidate.get("record") or {}
            kind = matched_candidate.get("kind")
            matched = matched_candidate.get("matched")

            if kind == "fake_hadith":

                if matched == "incorrect_hadith":
                    source_data = (
                        record.get("incorrect_hadith") or {}
                    )
                else:
                    source_data = (
                        record.get("correct_hadith") or {}
                    )

                source = {
                    "scholar": source_data.get(
                        "grader",
                        "",
                    ),
                    "title": source_data.get(
                        "source",
                        "",
                    ),
                    "url": source_data.get(
                        "reference_url",
                        "",
                    ),
                }

            elif kind == "hadith":

                source = {
                    "scholar": record.get(
                        "grader",
                        "",
                    ),
                    "title": record.get(
                        "source",
                        "",
                    ),
                    "url": record.get(
                        "reference_url",
                        "",
                    ),
                }

            elif kind == "fatwa":

                scholars = record.get(
                    "scholars"
                ) or []

                if isinstance(scholars, list):
                    scholar = ", ".join(
                        str(item)
                        for item in scholars
                    )
                else:
                    scholar = str(scholars)

                source = {
                    "scholar": scholar,
                    "title": record.get(
                        "title",
                        "",
                    ),
                    "url": record.get(
                        "reference_url",
                        "",
                    ),
                }

        return VerificationResponse(
            content_type=content_type,
            status=result.get(
                "status",
                "غير موثّق",
            ),
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