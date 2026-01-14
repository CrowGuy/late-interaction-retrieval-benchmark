# src/interactions/native.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.interactions.base import LateInteraction, ScoreBreakdown, ArtifactsLevel


@dataclass
class NativeScoreConfig:
    device: str = "cuda"  # where to run native scoring if torch needed
    prefer: str = "auto"  # "auto" | "processor" | "model"
    # If True: try to pass embeddings. If False: try raw inputs (not implemented here).
    embedding_only: bool = True


class NativeScoreInteraction(LateInteraction):
    """
    A compatibility interaction that calls model/processor-provided scoring APIs.

    Supports (capability detection):
      1) processor.score(q_emb, d_emb)                     # granite-like
      2) model.get_score(s)(q_emb, d_emb)                  # nemo-like
      3) processor.score_multi_vector([q_emb],[d_emb])     # colqwen3-like demos

    Notes:
    - This is NOT guaranteed to exist for every model.
    - It typically won't provide token-level artifacts.
    - Use this mainly for sanity-check / parity with official demos.
    """

    def __init__(self, *, model: Any = None, processor: Any = None, cfg: Optional[NativeScoreConfig] = None):
        self.model = model
        self.processor = processor
        self.cfg = cfg or NativeScoreConfig()

        if model is None and processor is None:
            raise ValueError("NativeScoreInteraction requires at least one of model or processor.")

    @property
    def name(self) -> str:
        return "native_score"

    def score(
        self,
        q_repr: Dict[str, Any],
        d_repr: Dict[str, Any],
        *,
        artifacts_level: ArtifactsLevel = "none",
        topk: int = 5,
    ) -> ScoreBreakdown:
        q = q_repr["emb"]
        d = d_repr["emb"]

        # Convert numpy -> torch (native APIs usually expect torch)
        q_t = self._to_torch(q)
        d_t = self._to_torch(d)

        # Some APIs expect batched / list format. We'll adapt per call.
        s = self._call_native(q_t, d_t)
        return ScoreBreakdown(score=float(s), artifacts=None, debug={"native_path": self._last_path})

    def _to_torch(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.cfg.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.cfg.device)
        raise TypeError(f"Unsupported embedding type: {type(x)}")

    def _call_native(self, q: torch.Tensor, d: torch.Tensor) -> float:
        """
        Returns scalar float score.
        """
        self._last_path = "none"

        # 1) Prefer processor path if requested/available
        if self.cfg.prefer in ("auto", "processor") and self.processor is not None:
            p = self.processor

            # (a) processor.score(q, d)  (granite-like)
            if hasattr(p, "score"):
                self._last_path = "processor.score"
                out = p.score(q, d)
                return self._to_scalar(out)

            # (b) processor.score_multi_vector([q],[d])  (colqwen3 demo-like)
            if hasattr(p, "score_multi_vector"):
                self._last_path = "processor.score_multi_vector"
                out = p.score_multi_vector([q], [d])
                # often returns (1,1) tensor
                return self._to_scalar(out)

        # 2) Model path (nemo-like)
        if self.cfg.prefer in ("auto", "model") and self.model is not None:
            m = self.model

            # model.get_score(q, d) / model.get_scores(qs, ds)
            if hasattr(m, "get_score"):
                self._last_path = "model.get_score"
                out = m.get_score(q, d)
                return self._to_scalar(out)

            if hasattr(m, "get_scores"):
                self._last_path = "model.get_scores"
                # expect batch/list; adapt to 1-item case
                out = m.get_scores(q[None, ...], d[None, ...]) if q.ndim == 2 else m.get_scores(q, d)
                return self._to_scalar(out)

        raise RuntimeError(
            "No supported native scoring API found. "
            "Tried processor.score / processor.score_multi_vector / model.get_score / model.get_scores."
        )

    @staticmethod
    def _to_scalar(out: Any) -> float:
        if isinstance(out, (float, int)):
            return float(out)
        if isinstance(out, np.ndarray):
            return float(out.reshape(-1)[0])
        if isinstance(out, torch.Tensor):
            return float(out.reshape(-1)[0].item())
        # sometimes list of tensors
        if isinstance(out, list) and len(out) > 0:
            return NativeScoreInteraction._to_scalar(out[0])
        raise TypeError(f"Cannot convert output to scalar: {type(out)}")