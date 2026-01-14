# core/interactions/maxsim_torch.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch

from src.interactions.base import LateInteraction, ScoreBreakdown, InteractionArtifacts, ArtifactsLevel


def _as_float32_2d_np(x: Any, name: str) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D (L,D). Got shape={x.shape}")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def _as_bool_1d(mask: Any, L: int, name: str) -> np.ndarray:
    if mask is None:
        return np.ones((L,), dtype=bool)
    if isinstance(mask, list):
        mask = np.array(mask)
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be list or np.ndarray, got {type(mask)}")
    mask = mask.reshape(-1)
    if mask.shape[0] != L:
        raise ValueError(f"{name} length mismatch: expected {L}, got {mask.shape[0]}")
    if mask.dtype != bool:
        mask = mask.astype(np.int32) != 0
    return mask


def _pick_mask(meta: Dict[str, Any], candidates: Tuple[str, ...]) -> Optional[Any]:
    for k in candidates:
        if k in meta and meta[k] is not None:
            return meta[k]
    return None


def _chunk_aggregate(
    token_scores: np.ndarray,          # (Lq,)
    token_to_doc_idx: np.ndarray,      # (Lq,)
    q_mask: np.ndarray,                # (Lq,)
    d_meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    chunks = d_meta.get("chunks")
    if not chunks:
        return None

    starts: List[int] = []
    ends: List[int] = []
    ids: List[Any] = []
    for c in chunks:
        if not all(k in c for k in ("chunk_id", "start", "end")):
            raise ValueError("Each chunk must have keys: chunk_id/start/end")
        ids.append(c["chunk_id"])
        starts.append(int(c["start"]))
        ends.append(int(c["end"]))

    starts_a = np.array(starts, dtype=np.int32)
    ends_a = np.array(ends, dtype=np.int32)

    valid = q_mask & (token_to_doc_idx >= 0)
    doc_pos = token_to_doc_idx.copy()
    doc_pos[~valid] = -1

    in_chunk = (doc_pos[:, None] >= starts_a[None, :]) & (doc_pos[:, None] < ends_a[None, :])
    chunk_idx = np.where(in_chunk.any(axis=1), in_chunk.argmax(axis=1), -1).astype(np.int32)

    chunk_scores: Dict[Any, float] = {cid: 0.0 for cid in ids}
    chunk_hit_counts: Dict[Any, int] = {cid: 0 for cid in ids}

    for i in np.where(valid)[0]:
        ci = int(chunk_idx[i])
        if ci < 0:
            continue
        cid = ids[ci]
        chunk_scores[cid] += float(token_scores[i])
        chunk_hit_counts[cid] += 1

    ranked = sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_chunks = ranked[: min(10, len(ranked))]

    return {
        "chunk_scores": chunk_scores,
        "chunk_hit_counts": chunk_hit_counts,
        "top_chunks": top_chunks,
        "token_to_chunk": [(ids[int(ci)] if int(ci) >= 0 else None) for ci in chunk_idx.tolist()],
    }


class MaxSimTorch(LateInteraction):
    """
    Torch/GPU block-wise MaxSim (no full sim matrix) + masking + optional topk artifacts.

    Optimization added:
      - When artifacts_level == "topk": compute topk only for valid query tokens
        (as indicated by q_mask), and scatter results back to full (Lq,K).
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        assume_normalized: bool = False,
        block_size: int = 2048,
        neg_value: float = -1e9,
        matmul_dtype: Optional[torch.dtype] = None,
    ):
        self.device = torch.device(device)
        self.assume_normalized = assume_normalized
        self.block_size = int(block_size)
        self.neg_value = float(neg_value)
        self.matmul_dtype = matmul_dtype  # e.g. torch.float16 / torch.bfloat16

        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")

    @property
    def name(self) -> str:
        return "maxsim_torch_blockwise"

    @torch.inference_mode()
    def score(
        self,
        q_repr: Dict[str, Any],
        d_repr: Dict[str, Any],
        *,
        artifacts_level: ArtifactsLevel = "none",
        topk: int = 5,
    ) -> ScoreBreakdown:
        # --- load numpy ---
        q_np = _as_float32_2d_np(q_repr["emb"], "q_repr['emb']")  # (Lq,D)
        d_np = _as_float32_2d_np(d_repr["emb"], "d_repr['emb']")  # (Ld,D)

        q_meta = q_repr.get("meta") or {}
        d_meta = d_repr.get("meta") or {}

        q_mask_raw = _pick_mask(q_meta, ("attention_mask", "query_mask", "mask"))
        d_mask_raw = _pick_mask(d_meta, ("image_mask", "attention_mask", "doc_mask", "mask"))

        q_mask = _as_bool_1d(q_mask_raw, q_np.shape[0], "q_mask")
        d_mask = _as_bool_1d(d_mask_raw, d_np.shape[0], "d_mask")

        Lq, D = q_np.shape
        Ld, D2 = d_np.shape
        if D != D2:
            raise ValueError(f"Dim mismatch: q dim {D} vs d dim {D2}")

        # --- to torch ---
        q = torch.from_numpy(q_np).to(self.device)
        d = torch.from_numpy(d_np).to(self.device)

        if self.matmul_dtype is not None:
            q = q.to(self.matmul_dtype)
            d = d.to(self.matmul_dtype)

        # normalize (optional)
        if not self.assume_normalized:
            q = torch.nn.functional.normalize(q, p=2, dim=1)
            d = torch.nn.functional.normalize(d, p=2, dim=1)

        # masks to torch
        q_mask_t = torch.from_numpy(q_mask.astype(np.bool_)).to(self.device)  # (Lq,)
        d_mask_t = torch.from_numpy(d_mask.astype(np.bool_)).to(self.device)  # (Ld,)

        neg = torch.tensor(self.neg_value, device=self.device, dtype=q.dtype)

        # best per query token (argmax path)
        best_scores = torch.full((Lq,), neg.item(), device=self.device, dtype=q.dtype)
        best_idx = torch.full((Lq,), -1, device=self.device, dtype=torch.int32)

        need_topk = (artifacts_level == "topk")
        if need_topk:
            K = max(1, int(topk))

            valid_rows = torch.where(q_mask_t)[0]  # (N_valid,)
            N_valid = int(valid_rows.numel())

            # allocate only for valid rows to save compute/memory
            topk_scores_v = torch.full((N_valid, K), neg.item(), device=self.device, dtype=q.dtype)
            topk_idx_v = torch.full((N_valid, K), -1, device=self.device, dtype=torch.int32)

            q_valid = q.index_select(0, valid_rows)  # (N_valid, D)

        bs = self.block_size

        # block-wise scan over doc tokens
        for start in range(0, Ld, bs):
            end = min(Ld, start + bs)

            d_block = d[start:end]                       # (B,D)
            dmask_block = d_mask_t[start:end]            # (B,)
            B = end - start

            # ---------- argmax (all rows) ----------
            sim_all = q @ d_block.transpose(0, 1)        # (Lq,B)
            if (~dmask_block).any():
                sim_all[:, ~dmask_block] = neg

            block_max, block_arg = torch.max(sim_all, dim=1)
            better = block_max > best_scores
            if better.any():
                best_scores = torch.where(better, block_max, best_scores)
                best_idx = torch.where(better, (start + block_arg).to(torch.int32), best_idx)

            # ---------- topk (valid rows only) ----------
            if need_topk and N_valid > 0:
                # compute sim only for valid rows
                sim_v = q_valid @ d_block.transpose(0, 1)   # (N_valid, B)
                if (~dmask_block).any():
                    sim_v[:, ~dmask_block] = neg

                k_block = min(K, B)
                blk_vals, blk_arg = torch.topk(sim_v, k=k_block, dim=1, largest=True, sorted=True)
                blk_idx = (start + blk_arg).to(torch.int32)

                if k_block < K:
                    pad_scores = torch.full((N_valid, K - k_block), neg.item(), device=self.device, dtype=q.dtype)
                    pad_idx = torch.full((N_valid, K - k_block), -1, device=self.device, dtype=torch.int32)
                    blk_vals = torch.cat([blk_vals, pad_scores], dim=1)
                    blk_idx = torch.cat([blk_idx, pad_idx], dim=1)

                # merge current (N_valid,K) with block (N_valid,K) -> keep top K
                cand_scores = torch.cat([topk_scores_v, blk_vals], dim=1)  # (N_valid,2K)
                cand_idx = torch.cat([topk_idx_v, blk_idx], dim=1)

                new_vals, sel = torch.topk(cand_scores, k=K, dim=1, largest=True, sorted=True)
                new_idx = torch.gather(cand_idx, 1, sel)
                topk_scores_v, topk_idx_v = new_vals, new_idx

        # apply query mask for argmax outputs
        best_scores = torch.where(q_mask_t, best_scores, torch.zeros_like(best_scores))
        best_idx = torch.where(q_mask_t, best_idx, torch.full_like(best_idx, -1))
        score = float(best_scores.sum().item())

        if artifacts_level == "none":
            return ScoreBreakdown(score=score)

        # move small results back to cpu numpy
        token_scores_np = best_scores.detach().to(torch.float32).cpu().numpy()
        token_to_doc_idx_np = best_idx.detach().cpu().numpy().astype(np.int32)

        stats: Dict[str, Any] = {
            "Lq": int(Lq),
            "Ld": int(Ld),
            "valid_q": int(q_mask.sum()),
            "valid_d": int(d_mask.sum()),
            "block_size": int(self.block_size),
            "backend": "torch",
            "device": str(self.device),
            "dtype": str(q.dtype).replace("torch.", ""),
        }

        chunk_info = _chunk_aggregate(token_scores_np, token_to_doc_idx_np, q_mask, d_meta)
        if chunk_info is not None:
            stats.update(chunk_info)

        # include token distribution stats (after masking)
        if q_mask.any():
            stats["token_score_mean"] = float(token_scores_np[q_mask].mean())
            stats["token_score_std"] = float(token_scores_np[q_mask].std())
        else:
            stats["token_score_mean"] = 0.0
            stats["token_score_std"] = 0.0

        if artifacts_level == "token":
            artifacts = InteractionArtifacts(
                token_scores=token_scores_np,
                token_to_doc_idx=token_to_doc_idx_np,
                stats=stats,
            )
            return ScoreBreakdown(score=score, artifacts=artifacts)

        # artifacts_level == "topk": scatter valid rows back to full (Lq,K)
        K = max(1, int(topk))
        topk_scores_full = torch.zeros((Lq, K), device=self.device, dtype=q.dtype)
        topk_idx_full = torch.full((Lq, K), -1, device=self.device, dtype=torch.int32)

        if q_mask_t.any():
            valid_rows = torch.where(q_mask_t)[0]
            # topk_scores_v/topk_idx_v are already (N_valid,K)
            topk_scores_full.index_copy_(0, valid_rows, topk_scores_v)
            topk_idx_full.index_copy_(0, valid_rows, topk_idx_v)

        topk_scores_np = topk_scores_full.detach().to(torch.float32).cpu().numpy()
        topk_idx_np = topk_idx_full.detach().cpu().numpy().astype(np.int32)

        artifacts = InteractionArtifacts(
            token_scores=token_scores_np,
            token_to_doc_idx=token_to_doc_idx_np,
            topk_doc_idx=topk_idx_np,
            topk_scores=topk_scores_np,
            stats=stats,
        )
        return ScoreBreakdown(score=score, artifacts=artifacts)