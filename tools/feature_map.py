from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np

ROLE_EXEC = "exec"
ROLE_VER = "ver"


class MiniLMFeatureMap:
    """Frozen feature map for task–executor quality features.

    Context and API identity are serialized into one text string. The
    transformer is frozen; only the online linear head in quality.py is updated.
    Verifier correctness is not predicted, so verifier MiniLM features are not built.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "Install sentence-transformers to use MiniLM features: pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[Tuple[str, str, str], np.ndarray] = {}

    @staticmethod
    def _context_to_text(context: Any) -> str:
        if isinstance(context, str):
            return context
        return json.dumps(context, sort_keys=True, ensure_ascii=True)

    def service_text(self, context: Any, role: str, api_id: str) -> str:
        context_text = self._context_to_text(context)
        return f"role: {role}\napi: {api_id}\ncontext:\n{context_text}"

    def encode(self, context: Any, role: str, api_id: str) -> np.ndarray:
        context_text = self._context_to_text(context)
        key = (role, api_id, context_text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        text = f"role: {role}\napi: {api_id}\ncontext:\n{context_text}"
        vector = np.asarray(self.model.encode(text, convert_to_numpy=True), dtype=float).reshape(-1)
        if self.normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        self._cache[key] = vector
        return vector.copy()
