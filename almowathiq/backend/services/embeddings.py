"""
البحث بالمعنى (الجزء الدلالي من الـHybrid Retrieval).

- يحوّل النص إلى متجه (Embedding) باستخدام Gemini.
- يبني فهرسًا متجهيًا لكل نوع (hadith / fatwa) باستخدام FAISS،
  وإن لم يتوفر FAISS يستخدم numpy بنفس النتيجة (مناسب لبيانات صغيرة).
- متجهات قاعدة المعرفة تُحسب مرة واحدة وتُحفظ في .cache/ ،
  وتُعاد تلقائيًا فقط إذا تغيّرت البيانات أو النموذج.

التشابه بالمعنى يجد "مرشحين" فقط، ولا يعني أبدًا أن النص موثّق.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
from google.genai import types

from services.gemini_client import get_client

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM = 768
_BATCH = 50
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


def _embed_one_call(client, contents) -> list:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return [e.values for e in result.embeddings]


def embed_texts(texts: list) -> np.ndarray:
    """
    يرجع مصفوفة (عدد النصوص × 768) بعد التطبيع. يرفع استثناء عند الفشل.

    ملاحظة: gemini-embedding-2 نموذج متعدد الوسائط، وقد يدمج قائمة النصوص
    في متجه واحد. لذلك نتحقق من العدد، وإن لم يطابق نرسل كل نص وحده.
    """
    client = get_client()
    vectors = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        batch_vectors = _embed_one_call(client, batch)
        if len(batch_vectors) != len(batch):
            batch_vectors = [_embed_one_call(client, t)[0] for t in batch]
        vectors.extend(batch_vectors)

    if len(vectors) != len(texts):
        raise RuntimeError(f"Embeddings count mismatch: {len(vectors)} != {len(texts)}")

    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


CACHE_VERSION = 2  # يُرفع عند تغيير طريقة الحساب لإبطال الكاش القديم


def _fingerprint(texts: list) -> str:
    payload = json.dumps([CACHE_VERSION, EMBED_MODEL, EMBED_DIM, texts], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class VectorIndex:
    """فهرس متجهي بسيط: FAISS إن توفر، وإلا numpy."""

    def __init__(self, vectors: np.ndarray):
        self.size = len(vectors)
        self.backend = "faiss" if FAISS_AVAILABLE else "numpy"
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(vectors.shape[1])  # Inner product = cosine بعد التطبيع
            self._index.add(vectors)
        else:
            self._vectors = vectors

    def search(self, query_vec: np.ndarray, k: int) -> list:
        """يرجع [(position, cosine), ...] مرتبة تنازليًا."""
        k = min(k, self.size)
        q = query_vec.reshape(1, -1).astype("float32")
        if FAISS_AVAILABLE:
            scores, idx = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        sims = self._vectors @ q[0]
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


def build_index(name: str, texts: list, embed_fn=embed_texts) -> VectorIndex:
    """يبني الفهرس، ويستخدم الكاش إذا لم تتغير النصوص."""
    CACHE_DIR.mkdir(exist_ok=True)
    fp = _fingerprint(texts)
    cache_file = CACHE_DIR / f"{name}_{fp}.npy"

    vectors = np.load(cache_file) if cache_file.exists() else None
    if vectors is not None and len(vectors) == len(texts):
        logger.info("Embeddings: loaded %s from cache (%d vectors)", name, len(vectors))
    else:
        vectors = embed_fn(texts)
        for old in CACHE_DIR.glob(f"{name}_*.npy"):
            old.unlink()
        np.save(cache_file, vectors)
        logger.info("Embeddings: built %s (%d vectors) and cached", name, len(vectors))

    return VectorIndex(vectors)
