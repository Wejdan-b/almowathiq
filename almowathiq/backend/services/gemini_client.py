import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# اسم النموذج قابل للتغيير من .env بدون تعديل الكود
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY غير موجود في ملف .env")
        _client = genai.Client(api_key=api_key)
    return _client