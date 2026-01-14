# src/interactions/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

import numpy as np

ArtifactsLevel = Literal["none", "token", "topk"]


@dataclass(frozen=True)
class InteractionArtifacts:
    """
    Analysis-friendly evidence for failure case diagnosis.
    Keep it compact by default (do NOT store full sim matrix unless you really need it).
    """
    token_scores: Optional[np.ndarray] = None        # (Lq,)
    token_to_doc_idx: Optional[np.ndarray] = None    # (Lq,)
    topk_doc_idx: Optional[np.ndarray] = None        # (Lq, K)
    topk_scores: Optional[np.ndarray] = None         # (Lq, K)
    stats: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    artifacts: Optional[InteractionArtifacts] = None
    debug: Optional[Dict[str, Any]] = None


class LateInteraction(ABC):
    """
    Contract:
      score(q_repr, d_repr) -> ScoreBreakdown
    where q_repr/d_repr come from encoders and must include:
      - q_repr["emb"]: np.ndarray (Lq, D)
      - d_repr["emb"]: np.ndarray (Ld, D)
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def score(
        self,
        q_repr: Dict[str, Any],
        d_repr: Dict[str, Any],
        *,
        artifacts_level: ArtifactsLevel = "none",
        topk: int = 5,
    ) -> ScoreBreakdown: ...