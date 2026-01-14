# core/encoders/col_encoder.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import torch

from src.config.encoders import EncoderConfig
from src.encoders.loader import load_model_and_processor

Modality = Literal["text", "image", "image+text"]


class ColEncoder:
    """
    A unified encoder that supports multiple backends:

    1) colpali-engine style:
       - processor.process_queries([...]) / processor.process_images([...])
       - model(**batch) -> output (may be tensor or obj/dict)
       - optional: processor.get_image_mask(batch) -> (B, L)

    2) Tomoro ColQwen3 style:
       - processor.process_texts(texts=[...]) / processor.process_images(images=[...])
       - model(**batch) -> out.embeddings

    3) NemoRetriever style:
       - model.forward_queries([...]) / model.forward_images([...])  (processor can be None)

    Output is always:
      {"emb": np.ndarray float32 (L, D), "meta": {...}}

    Masking support:
      - query meta includes "attention_mask" if available (shape (L,))
      - doc meta includes "image_mask" if available (shape (L,)) else "attention_mask" if available
    """

    def __init__(self, cfg: EncoderConfig, *, device_map: Any = None):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.device_map = device_map if device_map is not None else self.device

        self.model, self.processor = load_model_and_processor(cfg, self.device_map)

        self._dim: Optional[int] = None

    @property
    def name(self) -> str:
        return f"col-encoder:{self.cfg.model_kind}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("dim unknown until first encode")
        return self._dim

    # ---------- public API ----------

    @torch.no_grad()
    def encode_query(self, *, text: str, **kwargs) -> Dict[str, Any]:
        x_bld, meta_b = self._forward_queries([text], **kwargs)
        x_ld = x_bld[0]
        meta_1 = self._slice_meta(meta_b, 0)

        self._set_dim(x_ld)

        meta: Dict[str, Any] = {
            "modality": "text",
            "text": text,
            "encoder": self.name,
            "dim": self._dim,
            **meta_1,
        }
        return {"emb": self._to_numpy_ld(x_ld), "meta": meta}

    @torch.no_grad()
    def encode_doc(self, *, image: Any, doc_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        x_bld, meta_b = self._forward_images([image], **kwargs)
        x_ld = x_bld[0]
        meta_1 = self._slice_meta(meta_b, 0)

        self._set_dim(x_ld)

        meta: Dict[str, Any] = {
            "modality": "image",
            "doc_id": doc_id,
            "encoder": self.name,
            "dim": self._dim,
            **meta_1,
        }
        return {"emb": self._to_numpy_ld(x_ld), "meta": meta}

    @torch.no_grad()
    def encode_queries(self, texts: Sequence[str], batch_size: int = 8, **kwargs) -> List[Dict[str, Any]]:
        outs: List[Dict[str, Any]] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            x_bld, meta_b = self._forward_queries(batch_texts, **kwargs)

            for i, (t, x_ld) in enumerate(zip(batch_texts, x_bld)):
                self._set_dim(x_ld)
                meta_1 = self._slice_meta(meta_b, i)
                outs.append(
                    {
                        "emb": self._to_numpy_ld(x_ld),
                        "meta": {
                            "modality": "text",
                            "text": t,
                            "encoder": self.name,
                            "dim": self._dim,
                            **meta_1,
                        },
                    }
                )
        return outs

    @torch.no_grad()
    def encode_docs(
        self,
        images: Sequence[Any],
        doc_ids: Optional[Sequence[str]] = None,
        batch_size: int = 4,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        outs: List[Dict[str, Any]] = []
        for start in range(0, len(images), batch_size):
            batch_imgs = list(images[start : start + batch_size])
            batch_ids = list(doc_ids[start : start + batch_size]) if doc_ids is not None else [None] * len(batch_imgs)

            x_bld, meta_b = self._forward_images(batch_imgs, **kwargs)

            for i, (did, x_ld) in enumerate(zip(batch_ids, x_bld)):
                self._set_dim(x_ld)
                meta_1 = self._slice_meta(meta_b, i)
                outs.append(
                    {
                        "emb": self._to_numpy_ld(x_ld),
                        "meta": {
                            "modality": "image",
                            "doc_id": did,
                            "encoder": self.name,
                            "dim": self._dim,
                            **meta_1,
                        },
                    }
                )
        return outs

    # ---------- internal forward paths ----------

    def _forward_queries(self, texts: List[str], **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
          x: torch.Tensor (B, L, D)
          meta_batched: dict where values may be batched arrays (B, L) etc.
        """
        # NemoRetriever-style (no processor required)
        if hasattr(self.model, "forward_queries"):
            x = self.model.forward_queries(texts, batch_size=len(texts))
            x = self._ensure_BLD(x).to(self.device)
            # no native masks available unless model exposes them
            return x, {}

        if self.processor is None:
            raise RuntimeError("Processor is None but model does not provide forward_queries().")

        batch = self._process_texts(texts)
        batch = self._move_features_to_device(batch, self.device)

        out = self.model(**batch)
        x = self._extract_embeddings(out)
        x = self._ensure_BLD(x)

        meta_b: Dict[str, Any] = {}
        if isinstance(batch, dict):
            if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
                meta_b["input_ids"] = batch["input_ids"].detach().cpu().numpy()  # (B, L)
            if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
                meta_b["attention_mask"] = batch["attention_mask"].detach().cpu().numpy()  # (B, L)

        return x, meta_b

    def _forward_images(self, images: List[Any], **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
          x: torch.Tensor (B, L, D)
          meta_batched: dict where values may be batched arrays (B, L) etc.
        """
        # NemoRetriever-style (no processor required)
        if hasattr(self.model, "forward_images"):
            x = self.model.forward_images(images, batch_size=len(images))
            x = self._ensure_BLD(x).to(self.device)
            return x, {}

        if self.processor is None:
            raise RuntimeError("Processor is None but model does not provide forward_images().")

        feats = self._process_images(images)
        feats = self._move_features_to_device(feats, self.device)

        out = self.model(**feats)
        x = self._extract_embeddings(out)
        x = self._ensure_BLD(x)

        meta_b: Dict[str, Any] = {}

        # Some processors provide attention_mask for multimodal inputs
        if isinstance(feats, dict) and "attention_mask" in feats and isinstance(feats["attention_mask"], torch.Tensor):
            meta_b["attention_mask"] = feats["attention_mask"].detach().cpu().numpy()  # (B, L)

        # colpali-engine helper: image mask over sequence positions (best for doc-side masking)
        if self.processor is not None and hasattr(self.processor, "get_image_mask"):
            try:
                im = self.processor.get_image_mask(feats)  # expected torch.Tensor (B, L)
                if isinstance(im, torch.Tensor):
                    meta_b["image_mask"] = im.detach().cpu().numpy()
            except Exception:
                # not fatal; just skip
                pass

        return x, meta_b

    # ---------- processor adapters ----------

    def _process_texts(self, texts: List[str]) -> Any:
        p = self.processor
        # colpali-engine
        if hasattr(p, "process_queries"):
            return p.process_queries(texts)
        # tomoro-colqwen3
        if hasattr(p, "process_texts"):
            return p.process_texts(texts=texts)
        # generic HF
        return p(text=texts, return_tensors="pt", padding=True, truncation=True)

    def _process_images(self, images: List[Any]) -> Any:
        p = self.processor
        # colpali-engine + tomoro-colqwen3 both use process_images
        if hasattr(p, "process_images"):
            return p.process_images(images=images)
        # generic HF
        return p(images=images, return_tensors="pt")

    # ---------- output extraction & shape normalization ----------

    def _extract_embeddings(self, out: Any) -> torch.Tensor:
        """
        Supports:
          - out.embeddings
          - out.last_hidden_state
          - dict variants
          - direct tensor
        """
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "embeddings"):
            return out.embeddings
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, dict):
            if "embeddings" in out:
                return out["embeddings"]
            if "last_hidden_state" in out:
                return out["last_hidden_state"]
        raise RuntimeError("Cannot extract embeddings from model output. Please add a new extractor branch.")

    def _ensure_BLD(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize to (B, L, D).
        - (B, D) -> (B, 1, D)
        - (L, D) -> (1, L, D)  (for some single-item APIs)
        - (D,)   -> (1, 1, D)
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        if x.ndim == 3:
            return x

        if x.ndim == 2:
            # Heuristic:
            # If caller always uses batched paths, it's (B, D) and we treat L=1.
            # If some model returns (L, D) for single item, we also want (1, L, D).
            # We can disambiguate weakly by checking if second dim matches known dim.
            # But dim is unknown early. So:
            # - If row count is "small" and looks like token length, it's likely (L, D).
            # - If row count equals batch size, it's (B, D).
            # Here we default to (1, L, D) ONLY when we see a single item usage pattern later.
            # In our code path, _forward_* expects batched output; so interpret as (B, D).
            return x[:, None, :]

        if x.ndim == 1:
            return x[None, None, :]

        raise ValueError(f"Unexpected embedding shape: {tuple(x.shape)}")

    # ---------- meta slicing (batched -> single item) ----------

    @staticmethod
    def _slice_meta(meta_batched: Dict[str, Any], i: int) -> Dict[str, Any]:
        """
        Convert batched meta (B, ...) into single-item meta (...).
        - For arrays with first dim == B, return meta[i]
        - Otherwise pass-through
        """
        out: Dict[str, Any] = {}
        for k, v in (meta_batched or {}).items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                # assume first dim is batch
                if v.shape[0] > i:
                    out[k] = v[i]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    # ---------- utils ----------

    def _set_dim(self, x_ld: torch.Tensor) -> None:
        if self._dim is None:
            self._dim = int(x_ld.shape[-1])

    def _to_numpy_ld(self, x_ld: torch.Tensor) -> np.ndarray:
        return x_ld.detach().to("cpu", dtype=torch.float32).numpy()

    @staticmethod
    def _move_features_to_device(features: Any, device: torch.device) -> Any:
        if isinstance(features, dict):
            out = {}
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device)
                else:
                    out[k] = v
            return out
        if hasattr(features, "to"):
            return features.to(device)
        return features

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)