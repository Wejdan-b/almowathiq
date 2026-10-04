import logging

from google.genai import types
from pydantic import BaseModel

from services.gemini_client import generate_content, GEMINI_MODEL
from services.search import lexical_score as text_containment, normalize_arabic


logger = logging.getLogger(__name__)


class GeminiVerificationResult(BaseModel):
    same_text: bool
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

same_text:
true إذا كان نص الصورة هو نفس نص المرشح (حتى لو كان مقتطعًا،
أو تغيرت فيه كلمات، أو نُسب لعالم آخر).
false إذا كان نصًا آخر مختلفًا، حتى لو كان في نفس الموضوع أو قريبًا في المعنى.
إذا كانت same_text = false فاجعل is_match و is_truncated و is_altered كلها false.

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

    lexical_score = float(
        candidate.get("lexical_score", 0.0) or 0.0
    )

    def _correct_of_fake():
        correct = record.get("correct_hadith")
        if isinstance(correct, dict):
            correct = correct.get("text")
        return correct

    # حكم مباشر بـ"لا يصح" فقط إذا كان النص موجودًا حرفيًا.
    # أما التشابه بالمعنى وحده فيذهب لـGemini، حتى لا نحكم على حديث صحيح
    # قريب في المعنى (مثل "الطهور شطر الإيمان") بأنه لا يصح.
    semantic_score = candidate.get("semantic_score")
    semantic_score = float(semantic_score) if semantic_score is not None else 0.0

    if (
        kind == "fake_hadith"
        and matched == "incorrect_hadith"
        and (
            lexical_score >= 0.85
            or (lexical_score >= 0.6 and semantic_score >= 0.95)
        )
    ):
        return {
            "status": "يحتاج تصحيح",
            "issue": "not_authentic",
            "confidence": lexical_score,
            "matched_id": matched_id,
            "correct_text": _correct_of_fake(),
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

    # أسماء العلماء فقط (بدون تكرار) ليتحقق Gemini من النسبة
    if isinstance(scholars, list):
        names = []
        for item in scholars:
            name = item.get("name", "") if isinstance(item, dict) else str(item)
            if name and name not in names:
                names.append(name)
        scholars_text = "، ".join(names)
    else:
        scholars_text = str(scholars)

    if (
        kind == "fake_hadith"
        and matched == "correct_hadith"
        and lexical_score < 0.85
    ):
        return {
            "status": "غير موثّق",
            "issue": None,
            "confidence": lexical_score,
            "matched_id": None,
            "correct_text": None,
            "explanation": "لم تصل مطابقة النص إلى الحد المطلوب للتوثيق.",
        }

    # كلمات تقلب الحكم أو تغيّره: إذا اختلفت بين الصورة والمصدر فالنص محرّف
    # (التطابق الحرفي وحده لا يكشف حذف "لا" من "لا يحرم")
    def _ruling_words(text):
        critical = {
            "لا", "ليس", "ليست", "لم", "لن", "غير", "ما",
            "يجوز", "يحرم", "يجب", "يسن", "يستحب", "يكره", "يباح", "يشترط",
            "حرام", "حلال", "واجب", "سنه", "مكروه", "مباح", "جائز", "صحيح", "ضعيف",
        }
        words = normalize_arabic(text).split()
        return {w for w in words if w in critical}

    ruling_changed = _ruling_words(extracted_text) != (
        _ruling_words(candidate_text) & _ruling_words(extracted_text)
    ) or bool(
        # كلمة نفي/حكم في المصدر سقطت من الصورة رغم أن الصورة تغطي المصدر تقريبًا
        text_containment(candidate_text, extracted_text) >= 0.8
        and _ruling_words(candidate_text) - _ruling_words(extracted_text)
    )

    # التغطية: كم من نص المصدر موجود في نص الصورة (0..1)
    coverage = text_containment(candidate_text, extracted_text) if candidate_text else 0.0

    # نص الصورة موجود حرفيًا داخل المصدر، وطويل بما يكفي ليكون ذا معنى
    # => هو نفس النص (كامل أو مقتطع) مهما قال النموذج
    is_fragment_of_source = (
        lexical_score >= 0.85
        and len(normalize_arabic(extracted_text).replace(" ", "")) >= 20
    )

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

        is_fake_text = (
            kind == "fake_hadith" and matched == "incorrect_hadith"
        )

        same_text = result.same_text or is_fragment_of_source
        is_truncated = result.is_truncated or coverage < 0.8

        if not same_text:
            # نص آخر مختلف (حتى لو قريب في الموضوع): لا نتهمه بالتحريف
            status = "غير موثّق"
            issue = None

        elif is_fake_text:
            # نفس الحديث المنتشر الذي لا يصح (بصياغة قريبة)
            status = "يحتاج تصحيح"
            issue = "not_authentic"

        elif result.is_altered or ruling_changed:
            status = "يحتاج تصحيح"
            issue = "altered"

        elif is_truncated:
            # الاقتطاع يُحسب رياضيًا أيضًا (التغطية < 80%)، لا نعتمد على النموذج وحده
            status = "يحتاج تصحيح"
            issue = "truncated"

        elif (result.is_match or is_fragment_of_source) and lexical_score >= 0.85:
            status = "موثّق"
            issue = None

        else:
            status = "غير موثّق"
            issue = None

        if status == "غير موثّق":
            correct_text = None
            matched_id = None
        elif is_fake_text:
            correct_text = _correct_of_fake()
        else:
            correct_text = candidate_text

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