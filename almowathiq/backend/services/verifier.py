import logging

from google.genai import types
from pydantic import BaseModel

from services.gemini_client import generate_content, GEMINI_MODEL


logger = logging.getLogger(__name__)


class GeminiVerificationResult(BaseModel):
    is_match: bool
    is_truncated: bool
    is_altered: bool
    confidence: float
    explanation: str


PROMPT = """
أنت أداة تحقق سياقي للنصوص الدينية.

قارن النص المستخرج من الصورة مع بيانات المرشح التي أعطيت لك فقط.

ممنوع استخدام أي معرفة خارجية.
ممنوع اختراع نص أو مصدر أو عالم.
ممنوع إصدار حكم شرعي من عندك.

is_match:
true فقط إذا كان النص مطابقًا للنص الموجود في المرشح.

is_truncated:
true إذا كان النص الموجود في الصورة جزءًا من النص الأصلي
وتم حذف جزء منه.

is_altered:
true إذا تغيرت كلمات أو عبارات أو المعنى
أو كانت نسبة النص إلى عالم مختلف عن العالم الموجود في بيانات المرشح.

إذا كان النص مقتطعًا ومحرّفًا معًا اجعل is_altered = true.

confidence:
درجة الثقة في المقارنة النصية فقط من 0 إلى 1.

explanation:
شرح قصير باللغة العربية.
"""


def verify(extracted_text, content_type, candidates):

    if not extracted_text or not candidates:
        return {
            "status": "غير موثّق",
            "issue": None,
            "confidence": 0.0,
            "matched_id": None,
            "correct_text": None,
            "explanation": "لم يتم العثور على تطابق مناسب للتحقق.",
        }

    candidate = max(
        candidates,
        key=lambda item: float(item.get("score", 0.0)),
    )

    score = float(candidate.get("score", 0.0))

    if score < 0.45:
        return {
            "status": "غير موثّق",
            "issue": None,
            "confidence": 0.0,
            "matched_id": None,
            "correct_text": None,
            "explanation": "لم نجد تطابقًا موثوقًا في المصادر المستخدمة.",
        }

    record = candidate.get("record") or {}
    kind = candidate.get("kind")
    matched = candidate.get("matched")
    matched_id = candidate.get("id")

    if kind == "fake_hadith" and matched == "incorrect_hadith":

        correct = record.get("correct_hadith")

        if isinstance(correct, dict):
            correct = correct.get("text")

        return {
            "status": "يحتاج تصحيح",
            "issue": "not_authentic",
            "confidence": float(
                candidate.get("lexical_score", score)
            ),
            "matched_id": matched_id,
            "correct_text": correct,
            "explanation": (
                "النص الموجود في الصورة مطابق لحديث مسجل "
                "في قاعدة البيانات على أنه غير صحيح."
            ),
        }

    if kind == "fake_hadith":

        if matched == "incorrect_hadith":
            value = record.get("incorrect_hadith")
        else:
            value = record.get("correct_hadith")

        if isinstance(value, dict):
            candidate_text = value.get("text", "")
        else:
            candidate_text = str(value or "")

    else:
        candidate_text = str(record.get("text") or "")

    question = str(record.get("question") or "")
    scholars = record.get("scholars") or []

    if isinstance(scholars, list):
        scholars_text = ", ".join(
            str(item) for item in scholars
        )
    else:
        scholars_text = str(scholars)

    lexical_score = float(
        candidate.get("lexical_score", 0.0) or 0.0
    )

    if (
        kind == "fake_hadith"
        and matched == "correct_hadith"
        and lexical_score < 0.85
    ):
        return {
            "status": "غير موثّق",
            "issue": None,
            "confidence": lexical_score,
            "matched_id": matched_id,
            "correct_text": None,
            "explanation": "لم تصل مطابقة النص إلى الحد المطلوب للتوثيق.",
        }

    try:

        response = generate_content(
            model=GEMINI_MODEL,
            contents=[
                PROMPT,
                "النص المستخرج من الصورة:",
                extracted_text,
                "النص الموجود في المرشح:",
                candidate_text,
                "السؤال المرتبط بالفتوى إن وجد:",
                question,
                "العلماء المرتبطون بالمرشح إن وجدوا:",
                scholars_text,
                "نوع المحتوى:",
                content_type,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiVerificationResult,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        result = response.parsed

        if result is None:
            raise RuntimeError(
                "Gemini لم يرجع نتيجة تحقق صالحة"
            )

        confidence = max(
            0.0,
            min(
                1.0,
                float(result.confidence),
            ),
        )

        if result.is_altered:
            status = "يحتاج تصحيح"
            issue = "altered"

        elif result.is_truncated:
            status = "يحتاج تصحيح"
            issue = "truncated"

        elif result.is_match and lexical_score >= 0.85:
            status = "موثّق"
            issue = None

        else:
            status = "غير موثّق"
            issue = None

        if status in ["موثّق", "يحتاج تصحيح"]:
            correct_text = candidate_text
        else:
            correct_text = None

        return {
            "status": status,
            "issue": issue,
            "confidence": round(confidence, 2),
            "matched_id": matched_id,
            "correct_text": correct_text,
            "explanation": result.explanation,
        }

    except Exception as error:

        logger.exception(
            "Verifier Gemini error: %s",
            error,
        )

        return {
            "status": "غير موثّق",
            "issue": None,
            "confidence": 0.0,
            "matched_id": None,
            "correct_text": None,
            "explanation": (
                "تعذر إجراء التحقق حاليًا بسبب خطأ "
                "في خدمة التحقق."
            ),
        } 