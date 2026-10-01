from fastapi import FastAPI, UploadFile, File, HTTPException
from models.schemas import VerificationResponse


app = FastAPI(
    title="الموثّق الذكي API",
    description="Backend for Al-Mowathiq hackathon MVP",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "project": "الموثّق الذكي",
        "status": "running",
        "version": "0.1.0"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/verify",
    response_model=VerificationResponse
)
async def verify(
    image: UploadFile = File(...)
):
    """
    MVP Mock Endpoint

    حاليا لا نستخدم Gemini.
    نستقبل الصورة ونعيد نتيجة تجريبية
    للتأكد من أن الـAPI والـFlutter قادران على التواصل.
    """

    if not image.content_type:
        raise HTTPException(
            status_code=400,
            detail="نوع الملف غير معروف"
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="يسمح فقط بصور JPG أو PNG أو WEBP"
        )

    # نقرأ الصورة في الذاكرة فقط.
    # لن نحفظها على القرص في هذه المرحلة.
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="الصورة فارغة"
        )

    # ==============================
    # MOCK RESPONSE
    # ==============================

    return VerificationResponse(
        content_type="fatwa",
        status="موثّق",
        confidence=0.98,

        extracted_text=(
            "هذا نص تجريبي مستخرج من الصورة."
        ),

        correct_text=(
            "هذا نص تجريبي يمثل النص الكامل من المصدر."
        ),

        missing_context=[],

        explanation=(
            "هذه نتيجة تجريبية للتأكد من عمل الـAPI. "
            "سيتم استبدالها لاحقًا بنتيجة Gemini + البحث الدلالي."
        ),

        source={
            "scholar": "DEMO DATA",
            "title": "مصدر تجريبي — لا يمثل مصدرًا شرعيًا حقيقيًا",
            "url": "https://example.com"
        }
    )