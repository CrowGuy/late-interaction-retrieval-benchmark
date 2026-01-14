# src/encoders/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, List, Literal

import numpy as np

Modality = Literal["text", "image", "image+text"]
Repr = Dict[str, Any]  # {"emb": np.ndarray (L,D), "meta": {...}}


def _ensure_inputs(text: Optional[str], image: Optional[Any], modality: Modality) -> None:
    if modality == "text" and not text:
        raise ValueError("modality='text' requires text")
    if modality == "image" and image is None:
        raise ValueError("modality='image' requires image")
    if modality == "image+text" and (image is None or not text):
        raise ValueError("modality='image+text' requires both image and text")


class MultiModalEncoder(ABC):
    """
    Encoder layer contract:
      raw inputs -> late-interaction-ready embeddings (L, D) + meta.

    Output must be:
      {
        "emb": np.ndarray float32, shape (L, D),
        "meta": dict
      }
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode_query(
        self,
        *,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        modality: Modality = "text",
        **kwargs
    ) -> Repr: ...

    @abstractmethod
    def encode_doc(
        self,
        *,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        modality: Modality = "text",
        doc_id: Optional[str] = None,
        **kwargs
    ) -> Repr: ...

    # ---- convenience wrappers (recommended) ----
    def encode_text_query(self, text: str, **kwargs) -> Repr:
        return self.encode_query(text=text, modality="text", **kwargs)

    def encode_image_query(self, image: Any, **kwargs) -> Repr:
        return self.encode_query(image=image, modality="image", **kwargs)

    def encode_text_doc(self, text: str, doc_id: Optional[str] = None, **kwargs) -> Repr:
        return self.encode_doc(text=text, modality="text", doc_id=doc_id, **kwargs)

    def encode_image_doc(self, image: Any, doc_id: Optional[str] = None, **kwargs) -> Repr:
        return self.encode_doc(image=image, modality="image", doc_id=doc_id, **kwargs)

    # ---- optional batch APIs (good for indexing) ----
    def encode_queries(self, texts: Sequence[str], **kwargs) -> List[Repr]:
        return [self.encode_text_query(t, **kwargs) for t in texts]

    def encode_docs(self, images: Sequence[Any], doc_ids: Optional[Sequence[str]] = None, **kwargs) -> List[Repr]:
        if doc_ids is None:
            return [self.encode_image_doc(im, **kwargs) for im in images]
        return [self.encode_image_doc(im, doc_id=did, **kwargs) for im, did in zip(images, doc_ids)]