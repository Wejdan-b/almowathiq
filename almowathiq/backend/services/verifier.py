import logging
from difflib import SequenceMatcher

from google.genai import types
from pydantic import BaseModel

from services.gemini_client import generate_content, GEMINI_MODEL
from services.search import lexical_score as text_containment, normalize_arabic


logger = logging.getLogger(__name__)


class GeminiVerificationResult(BaseModel):
    same_text: bool
    is_match: bool
    is_truncated: bool
    omission_changes_meaning: bool
    is_altered: bool
    attribution_mismatch: bool
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
true إذا تغيرت كلمات أو عبارات أو المعنى في النص نفسه.
(النسبة إلى راوٍ أو كتاب أو عالم تُقيَّم في attribution_mismatch وليس هنا.)

attribution_mismatch:
true إذا نسبت الصورة النص إلى قائل أو راوٍ أو كتاب أو عالم يخالف بيانات المرشح، مثل:
- راوٍ مختلف (مثلًا "عن عائشة" والراوي في المرشح أنس بن مالك)،
  مع مراعاة أن الأسماء المذكورة داخل نص المرشح نفسه صحيحة النسبة.
- قائل مختلف (مثلًا "قال رسول الله ﷺ" والنص في المرشح قول صحابي أو تابعي).
- كتاب غير موجود في المصدر ولا في "وأخرجه أيضًا".
- عالم غير موجود في قائمة العلماء (للفتاوى).
false إذا لم تذكر الصورة أي نسبة، أو كانت النسبة مطابقة.

omission_changes_meaning:
true فقط إذا كان الجزء المحذوف من النص الأصلي يغيّر الحكم أو يقيّده أو يغيّر فهم النص
(مثل حذف شرط، أو استثناء، أو تفصيل، أو جواب السؤال).
false إذا كان المحذوف لا يغيّر الحكم (مثل حذف ذكر من قال به أو الأدلة أو الإسناد).
إذا لم يكن هناك حذف فاجعلها false.

إذا كان النص مقتطعًا ومحرّفًا معًا اجعل is_altered = true و is_truncated = true.

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

    def _fake_explanation():
        # شرح ثابت من بيانات المصدر نفسه، وليس من صياغة النموذج
        fake = record.get("incorrect_hadith") or {}
        grade = fake.get("grade") or "لا يصح"
        grade_source = fake.get("grade_source") or "المصدر"
        return (
            f"هذا النص منتشر، وحكم عليه في {grade_source}: «{grade}». "
            "وهذا بديل صحيح في معناه."
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
            "issues": ["not_authentic"],
            "confidence": lexical_score,
            "matched_id": matched_id,
            "correct_text": _correct_of_fake(),
            "explanation": _fake_explanation(),
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

    # بيانات النسبة للحديث (من السجل نفسه)
    if kind == "fake_hadith":
        hadith_meta = record.get("correct_hadith") or {}
    elif kind == "hadith":
        hadith_meta = record
    else:
        hadith_meta = {}
    also_in = hadith_meta.get("also_in") or []
    attribution_text = ""
    if hadith_meta:
        attribution_text = (
            f"الراوي: {hadith_meta.get('narrator', '')}\n"
            f"المصدر: {hadith_meta.get('source', '')} ({hadith_meta.get('number', '')})\n"
            f"وأخرجه أيضًا: {'، '.join(also_in) or 'لا يوجد'}\n"
            f"الحكم: {hadith_meta.get('grade', '')} - {hadith_meta.get('grader', '')}"
        )

    # فحص رياضي: هل تذكر الصورة كتاب حديث غير موجود في بيانات السجل؟
    def _book_mismatch():
        if not hadith_meta:
            return []
        books = {
            "البخاري": "البخاري", "مسلم": "مسلم", "الترمذي": "الترمذي",
            "ابو داود": "داود", "النسائي": "النسائي", "ابن ماجه": "ماجه",
            "احمد": "احمد", "البيهقي": "البيهقي", "البزار": "البزار",
            "ابن خزيمه": "خزيمه", "ابن حبان": "حبان", "الطبراني": "الطبراني",
            "الحاكم": "الحاكم", "مالك": "مالك", "الدارمي": "الدارمي",
        }
        image_norm = " " + normalize_arabic(extracted_text) + " "
        known = normalize_arabic(" ".join(
            [str(hadith_meta.get("source", "")), str(hadith_meta.get("grader", ""))]
            + [str(x) for x in also_in]
        ))
        wrong = []
        for name, key in books.items():
            if f" {normalize_arabic(name)} " in image_norm and key not in known:
                wrong.append(name)
        return wrong

    wrong_books = _book_mismatch()

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

    # هل فعل الحكم منفي أم لا؟ ("لا يحرم" مقابل "يحرم") بغض النظر عن طول النص
    def _negation_map(text):
        negations = {"لا", "ليس", "ليست", "لم", "لن", "غير"}
        verbs = {
            "يجوز", "يحرم", "يجب", "يسن", "يستحب", "يكره", "يباح", "يشترط", "تشترط",
            "يصح", "تصح", "يلزم", "يشرع",
        }
        words = normalize_arabic(text).split()
        result_map = {}
        for i, w in enumerate(words):
            if w in verbs:
                result_map.setdefault(w, set()).add(i > 0 and words[i - 1] in negations)
        return result_map

    _img_neg, _src_neg = _negation_map(extracted_text), _negation_map(candidate_text)
    negation_flipped = any(
        verb in _src_neg and not (_img_neg[verb] & _src_neg[verb])
        for verb in _img_neg
    )

    # حذف كلمة نفي كانت في المصدر = تحريف + اقتطاع (حُذفت كلمة من النص الأصلي)
    negation_removed = any(
        verb in _src_neg and True in _src_neg[verb] and True not in _img_neg[verb]
        for verb in _img_neg
    )

    ruling_changed = negation_flipped or _ruling_words(extracted_text) != (
        _ruling_words(candidate_text) & _ruling_words(extracted_text)
    ) or bool(
        # كلمة نفي/حكم في المصدر سقطت من الصورة رغم أن الصورة تغطي المصدر تقريبًا
        text_containment(candidate_text, extracted_text) >= 0.8
        and _ruling_words(candidate_text) - _ruling_words(extracted_text)
    )

    # مقارنة كلمة بكلمة بين المصدر والصورة (داخل الجزء المنقول فقط)
    def _word_key(w):
        # "والمالكيه" و"المالكيه" نفس الكلمة: الواو الملتصقة لا تُعد تحريفًا
        return w[1:] if w.startswith("و") and len(w) > 3 else w

    def _inner_diff():
        src_words = normalize_arabic(candidate_text).split()
        img_words = normalize_arabic(extracted_text).split()
        sm = SequenceMatcher(
            None,
            [_word_key(w) for w in src_words],
            [_word_key(w) for w in img_words],
            autojunk=False,
        )
        ops = sm.get_opcodes()
        equal = [op for op in ops if op[0] == "equal"]
        removed, added = [], []
        if not equal:
            return removed, added
        # نطاق الجزء المنقول في الصورة: من أول تطابق إلى آخر تطابق
        first_src, last_src = equal[0][1], equal[-1][2]
        first_img, last_img = equal[0][3], equal[-1][4]
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                continue
            inside_src = i1 >= first_src and i2 <= last_src
            inside_img = j1 >= first_img and j2 <= last_img
            if tag in ("delete", "replace") and inside_src and inside_img:
                removed.extend(src_words[i1:i2])
            if tag in ("insert", "replace") and inside_src and inside_img:
                added.extend(img_words[j1:j2])
        return removed, added

    removed_words, added_words = _inner_diff()

    # هل الجزء المحذوف من المصدر يحمل شرطًا أو استثناءً أو حكمًا؟
    def _omission_has_qualifier():
        qualifiers = {
            "الا", "بشرط", "شرط", "اذا", "اما", "لكن", "بخلاف", "عدا", "سوي",
            "يجوز", "يحرم", "يجب", "يسن", "يستحب", "يكره", "يباح", "يشترط", "تشترط",
            "لا", "ليس", "لم", "لن", "حرام", "حلال", "مكروه", "مباح", "واجب", "جائز",
            "الاباحه", "فالاصل", "الاصل", "يصح", "تصح", "يلزم",
        }
        image_words = set(normalize_arabic(extracted_text).split())
        missing = [
            w for w in normalize_arabic(candidate_text).split() if w not in image_words
        ]
        return any(w in qualifiers for w in missing)

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
                "بيانات نسبة الحديث في المرشح إن وجدت:",
                attribution_text or "لا يوجد",
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
        # كلمة ناقصة من وسط النص = مقتطع، وكلمة مضافة أو مستبدلة = محرّف
        is_truncated = result.is_truncated or coverage < 0.8 or bool(removed_words)
        is_altered = result.is_altered or ruling_changed or bool(added_words)
        is_misattributed = result.attribution_mismatch or bool(wrong_books)
        meaningful_omission = is_truncated and (
            result.omission_changes_meaning
            or _omission_has_qualifier()
            or bool(removed_words)  # حذف من داخل النص المنقول لا يُعد اختصارًا بريئًا
        )

        issues = []

        if not same_text:
            # نص آخر مختلف (حتى لو قريب في الموضوع): لا نتهمه بالتحريف
            status = "غير موثّق"

        elif is_fake_text:
            # نفس الحديث المنتشر الذي لا يصح (بصياغة قريبة)
            status = "يحتاج تصحيح"
            issues = ["not_authentic"]

        elif is_altered or is_misattributed:
            status = "يحتاج تصحيح"
            issues = (
                (["altered"] if is_altered else [])
                + (["misattributed"] if is_misattributed else [])
                + (["truncated"] if (is_truncated or negation_removed) else [])
            )

        elif meaningful_omission:
            # اقتطاع يغيّر الحكم أو الفهم (حذف شرط أو استثناء أو تفصيل)
            status = "يحتاج تصحيح"
            issues = ["truncated"]

        elif (result.is_match or is_fragment_of_source) and lexical_score >= 0.85:
            status = "موثّق"
            # مختصر بدون تغيير في الحكم: موثّق مع ملاحظة
            issues = ["abridged"] if is_truncated else []

        else:
            status = "غير موثّق"

        issue = issues[0] if issues else None

        def _final_explanation():
            if issue == "not_authentic":
                return _fake_explanation()
            text = result.explanation or ""
            if status != "غير موثّق":
                if removed_words:
                    text += " الكلمات المحذوفة من النص الأصلي: «" + "، ".join(removed_words) + "»."
                if added_words:
                    text += " الكلمات المضافة أو المغيّرة في الصورة: «" + "، ".join(added_words) + "»."
                if wrong_books:
                    text += (
                        " نُسب في الصورة إلى: «" + "، ".join(wrong_books) + "»، "
                        "وهو غير مذكور في مصادر هذا الحديث عندنا."
                    )
            return text.strip()

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
            "issues": issues,
            "confidence": round(confidence, 2),
            "matched_id": matched_id,
            "correct_text": correct_text,
            "explanation": _final_explanation(),
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