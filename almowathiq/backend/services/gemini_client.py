import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

# اسم النموذج قابل للتغيير من .env بدون تعديل الكود
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# نماذج احتياطية تُجرَّب إذا كان النموذج الأساسي مضغوطًا (503/429)
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash").split(",")
    if m.strip()
]

# أخطاء مؤقتة تستحق إعادة المحاولة
RETRYABLE_CODES = {429, 500, 503, 504}
RETRY_DELAYS = [1.0, 2.0]  # انتظار بين المحاولات (ثوانٍ)

# حد زمني لكل طلب إلى Gemini، حتى لا يعلق التطبيق إذا تأخر الرد
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY غير موجود في ملف .env")
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000),
        )
    return _client


def is_retryable(error: Exception) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if code in RETRYABLE_CODES:
        return True
    # انقطاع الاتصال لحظيًا (شبكة) يستحق إعادة المحاولة أيضًا
    if type(error).__name__ in {
        "ReadError", "ConnectError", "RemoteProtocolError", "WriteError",
        "ReadTimeout", "ConnectTimeout", "PoolTimeout", "ConnectionResetError",
        "ConnectionError",
    }:
        return True
    text = str(error).upper()
    return any(k in text for k in (
        "UNAVAILABLE", "RESOURCE_EXHAUSTED", "TIMEOUT", "TIMED OUT",
        "10054", "CONNECTION RESET", "FORCIBLY CLOSED", "CONNECTION ABORTED",
    ))


def with_retry(call, label: str = "gemini"):
    """ينفذ call() ويعيد المحاولة عند الأخطاء المؤقتة فقط."""
    last_error = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return call()
        except Exception as e:
            last_error = e
            if not is_retryable(e) or attempt == len(RETRY_DELAYS):
                raise
            delay = RETRY_DELAYS[attempt]
            logger.warning("%s: temporary error (%s), retry %d in %.0fs",
                           label, str(e)[:80], attempt + 1, delay)
            time.sleep(delay)
    raise last_error


def _config_for(model: str, config):
    """نماذج Gemini 2.x لا تدعم thinking_level، فنحذفه عند استخدامها كاحتياط."""
    if config is None or not model.startswith("gemini-2"):
        return config
    if getattr(config, "thinking_config", None) is None:
        return config
    return config.model_copy(update={"thinking_config": None})


def generate_content(model: str = None, contents=None, config=None):
    """
    بديل آمن لـ client.models.generate_content:
    1) يعيد المحاولة على النموذج الأساسي عند الضغط المؤقت.
    2) إذا استمر الفشل المؤقت، يجرب النماذج الاحتياطية.
    الأخطاء غير المؤقتة (مثل 400 أو 404) تُرفع فورًا بدون تجربة احتياط.
    """
    primary = model or GEMINI_MODEL
    models = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    last_error = None

    for i, m in enumerate(models):
        try:
            result = with_retry(
                lambda: get_client().models.generate_content(
                    model=m, contents=contents, config=_config_for(m, config)
                ),
                label=f"generate[{m}]",
            )
            if i > 0:
                logger.warning("Gemini: used fallback model %s", m)
            return result
        except Exception as e:
            last_error = e
            if not is_retryable(e):
                raise
            logger.warning("Gemini: model %s unavailable, trying next", m)

    raise last_error
