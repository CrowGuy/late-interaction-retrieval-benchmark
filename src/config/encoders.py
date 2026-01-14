# src/config/encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch

ModelKind = Literal["colpali", "colqwen2", "colqwen2_5", "auto"]


@dataclass
class EncoderConfig:
    """
    Pure configuration (no heavy imports).
    """
    model_name: str
    model_kind: ModelKind = "colpali"

    # Runtime
    device: str = "auto"  # e.g. "auto", "cuda", "cuda:0", "cpu"
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True

    # Optional model/processor kwargs (covers cases like Tomoro colqwen3)
    # Examples:
    #   processor_kwargs={"max_num_visual_tokens": 1280}
    #   model_kwargs={"attn_implementation": "flash_attention_2"}
    processor_kwargs: Optional[Dict] = None
    model_kwargs: Optional[Dict] = None


# Explicit aliases (highest priority)
MODEL_KIND_ALIASES: Dict[str, ModelKind] = {
    "colnomic-embed-multimodal-7b": "colqwen2_5",
}


def infer_model_kind(model_name: str) -> ModelKind:
    s = model_name.lower()

    for key, kind in MODEL_KIND_ALIASES.items():
        if key in s:
            return kind

    if any(k in s for k in ("colqwen2.5", "colqwen2_5", "colqwen2-5")):
        return "colqwen2_5"
    if "colqwen2" in s:
        return "colqwen2"
    if "colpali" in s:
        return "colpali"

    return "auto"