import time

from google.genai import types
from pydantic import BaseModel

from services.gemini_client import generate_content, GEMINI_MODEL


class OCRResult(BaseModel):
    is_readable: bool
    text: str


OCR_PROMPT = """You are a strict OCR engine. Transcribe the Arabic text visible in this image.

Rules:
1. Copy the text EXACTLY as it appears: same words, same order, same diacritics (tashkeel) if present, same punctuation.
2. Do NOT correct spelling, grammar, or typos.
3. Do NOT complete cut-off words or sentences, even if you recognize the text (for example a known hadith, verse, or fatwa). Write only what is visible.
4. Do NOT add, paraphrase, translate, explain, summarize, or comment.
5. Keep content attributions that appear in the image (for example: رواه البخاري, or a scholar's name).
6. Ignore everything that is not part of the religious content itself: app interface elements (clock, battery, like counts, share buttons), account names, usernames, page or channel names, watermarks, logos, hashtags, and promotional text.
   Keep attributions of the content to sources or scholars (for example: صحيح مسلم، رواه البخاري، قال الشيخ ...), because they are part of the claim being verified.
7. Keep line order. Separate lines with a newline.
8. If a specific word is unclear, write [غير واضح] in its place instead of guessing.
9. If the image has no readable Arabic text, or is too blurry to read reliably, set is_readable to false and text to an empty string.
"""


def extract_text(image_bytes: bytes, mime_type: str) -> dict:
    """يستخرج النص العربي من الصورة كما هو، بدون تصحيح أو إكمال."""
    start = time.perf_counter()
    response = generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            OCR_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OCRResult,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    elapsed = time.perf_counter() - start

    result = response.parsed
    if result is None:
        raise RuntimeError("Gemini لم يرجع نتيجة OCR صالحة")

    text = result.text.strip()
    readable = result.is_readable and bool(text)

    return {
        "text": text if readable else "",
        "is_readable": readable,
        "elapsed_seconds": round(elapsed, 2),
    }
