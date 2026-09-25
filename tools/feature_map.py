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
        del role  # Quality features are task–model pairs; the verifier is not a second head.
        return self.feature_text(context, api_id)

    @staticmethod
    def feature_text(context: Any, api_id: str) -> str:
        """Paper A.3.3 / Sec. 3.3 feature string.

        Encodes the node description, task instruction, required upstream outputs,
        depth and in-/out-degree, and the candidate model description.
        """
        if not isinstance(context, dict):
            return f"model: {api_id}\n{context}"
        graph = context.get("graph_features") if isinstance(context.get("graph_features"), dict) else {}
        predecessors = context.get("predecessors") or []
        if isinstance(predecessors, list):
            required = ", ".join(str(item) for item in predecessors) if predecessors else "none"
        else:
            required = str(predecessors)
        return (
            f"task: {context.get('task_description', '')}\n"
            f"instruction: {context.get('input_template', '')}\n"
            f"required_outputs: {required}\n"
            f"depth: {graph.get('depth', 0)}\n"
            f"in_degree: {graph.get('in_degree', 0)}\n"
            f"out_degree: {graph.get('out_degree', 0)}\n"
            f"model: {api_id}"
        )

    def encode(self, context: Any, role: str, api_id: str) -> np.ndarray:
        del role
        context_text = self.feature_text(context, api_id)
        key = ("task_model", api_id, context_text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        text = context_text
        vector = np.asarray(self.model.encode(text, convert_to_numpy=True), dtype=float).reshape(-1)
        if self.normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        self._cache[key] = vector
        return vector.copy()
