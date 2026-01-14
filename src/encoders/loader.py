# src/encoders/loader.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from src.config.encoders import EncoderConfig, ModelKind

# Registry is lazy: each entry returns (ModelCls, ProcessorCls_or_None)
MODEL_REGISTRY: Dict[ModelKind, Callable[[], Tuple[Any, Optional[Any]]]] = {}


def _register_defaults() -> None:
    """
    Populate MODEL_REGISTRY exactly once.
    Uses lazy import so importing this module doesn't require colpali_engine/transformers installed.
    """
    if MODEL_REGISTRY:
        return

    def load_colpali():
        from colpali_engine.models import ColPali, ColPaliProcessor
        return ColPali, ColPaliProcessor

    def load_colqwen2():
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        return ColQwen2, ColQwen2Processor

    def load_colqwen2_5():
        # NOTE: name may vary depending on your installed colpali-engine version
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        return ColQwen2_5, ColQwen2_5_Processor

    def load_auto():
        from transformers import AutoModel, AutoProcessor
        return AutoModel, AutoProcessor

    MODEL_REGISTRY.update({
        "colpali": load_colpali,
        "colqwen2": load_colqwen2,
        "colqwen2_5": load_colqwen2_5,
        "auto": load_auto,
    })


def load_model_and_processor(cfg: EncoderConfig, device_map: Any):
    """
    Returns (model, processor_or_none).

    - For colpali-engine kinds: returns processor instance.
    - For 'auto': returns AutoProcessor instance by default (but still optional in caller logic).
    """
    _register_defaults()

    if cfg.model_kind not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_kind={cfg.model_kind}. Supported: {list(MODEL_REGISTRY.keys())}")

    # Lazy import model/processor classes
    try:
        ModelCls, ProcCls = MODEL_REGISTRY[cfg.model_kind]()
    except ImportError as e:
        if cfg.model_kind in ("colpali", "colqwen2", "colqwen2_5"):
            raise ImportError(
                f"Failed to import colpali_engine for model_kind={cfg.model_kind}. "
                f"Please install colpali-engine. Original error: {e}"
            ) from e
        if cfg.model_kind == "auto":
            raise ImportError(
                f"Failed to import transformers for model_kind='auto'. "
                f"Please install transformers. Original error: {e}"
            ) from e
        raise

    # Processor kwargs
    processor = None
    if ProcCls is not None:
        proc_kwargs: Dict[str, Any] = {}
        if cfg.model_kind == "auto":
            proc_kwargs["trust_remote_code"] = cfg.trust_remote_code
        if cfg.processor_kwargs:
            proc_kwargs.update(cfg.processor_kwargs)
        processor = ProcCls.from_pretrained(cfg.model_name, **proc_kwargs)

    # Model kwargs
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": cfg.torch_dtype,
        "device_map": device_map,
    }
    if cfg.model_kind == "auto":
        model_kwargs["trust_remote_code"] = cfg.trust_remote_code
    if cfg.model_kwargs:
        model_kwargs.update(cfg.model_kwargs)

    model = ModelCls.from_pretrained(cfg.model_name, **model_kwargs).eval()
    return model, processor