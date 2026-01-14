# core/interactions/maxsim.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from src.interactions.base import LateInteraction, ScoreBreakdown, InteractionArtifacts, ArtifactsLevel


def _as_float32_2d(x: Any, name: str) -> np.ndarray:
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


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _merge_topk(
    cur_scores: np.ndarray,  # (Lq, K)
    cur_idx: np.ndarray,     # (Lq, K)
    new_scores: np.ndarray,  # (Lq, K)
    new_idx: np.ndarray,     # (Lq, K)
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge two (Lq,K) top-k sets into one (Lq,K) top-k set.
    """
    # (Lq, 2K)
    scores = np.concatenate([cur_scores, new_scores], axis=1)
    idx = np.concatenate([cur_idx, new_idx], axis=1)

    # select top K by score per row
    sel = np.argpartition(scores, -K, axis=1)[:, -K:]
    sel_scores = np.take_along_axis(scores, sel, axis=1)
    sel_idx = np.take_along_axis(idx, sel, axis=1)

    # sort descending within selected K
    order = np.argsort(sel_scores, axis=1)[:, ::-1]
    out_scores = np.take_along_axis(sel_scores, order, axis=1).astype(np.float32)
    out_idx = np.take_along_axis(sel_idx, order, axis=1).astype(np.int32)
    return out_scores, out_idx


class MaxSimNumpy(LateInteraction):
    """
    Block-wise MaxSim (memory safe for large Ld):

      score = sum_{i in valid_q} max_{j in valid_d} <q_i, d_j>

    Masking:
      - Query mask from q_meta: attention_mask/query_mask/mask
      - Doc mask from d_meta: image_mask (preferred) / attention_mask / doc_mask / mask

    artifacts_level:
      - "none": only score
      - "token": token_scores + argmax + (optional chunk aggregation in stats)
      - "topk": token-level top-k alignments (doc idx + score)

    Notes:
      - Does NOT materialize sim matrix (Lq,Ld); uses (Lq, block_size) chunks.
    """

    def __init__(
        self,
        *,
        assume_normalized: bool = False,
        block_size: int = 1024,
        neg_value: float = -1e9,
    ):
        self.assume_normalized = assume_normalized
        self.block_size = int(block_size)
        self.neg = np.float32(neg_value)

        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")

    @property
    def name(self) -> str:
        return "maxsim_blockwise"

    def score(
        self,
        q_repr: Dict[str, Any],
        d_repr: Dict[str, Any],
        *,
        artifacts_level: ArtifactsLevel = "none",
        topk: int = 5,
    ) -> ScoreBreakdown:
        q = _as_float32_2d(q_repr["emb"], "q_repr['emb']")  # (Lq, D)
        d = _as_float32_2d(d_repr["emb"], "d_repr['emb']")  # (Ld, D)

        q_meta = q_repr.get("meta") or {}
        d_meta = d_repr.get("meta") or {}

        q_mask_raw = _pick_mask(q_meta, ("attention_mask", "query_mask", "mask"))
        d_mask_raw = _pick_mask(d_meta, ("image_mask", "attention_mask", "doc_mask", "mask"))

        q_mask = _as_bool_1d(q_mask_raw, q.shape[0], "q_mask")
        d_mask = _as_bool_1d(d_mask_raw, d.shape[0], "d_mask")

        if not self.assume_normalized:
            qn = _l2_normalize_rows(q)
            dn = _l2_normalize_rows(d)
        else:
            qn, dn = q, d

        Lq = qn.shape[0]
        Ld = dn.shape[0]

        # init best per query token
        best_scores = np.full((Lq,), self.neg, dtype=np.float32)
        best_idx = np.full((Lq,), -1, dtype=np.int32)

        need_topk = (artifacts_level == "topk")
        if need_topk:
            K = max(1, int(topk))
            # global topk per query token
            topk_scores = np.full((Lq, K), self.neg, dtype=np.float32)
            topk_idx = np.full((Lq, K), -1, dtype=np.int32)

        # block-wise scan
        bs = self.block_size
        for start in range(0, Ld, bs):
            end = min(Ld, start + bs)
            d_block = dn[start:end]                # (B, D)
            dmask_block = d_mask[start:end]        # (B,)

            # compute sim for this block: (Lq, B)
            sim_block = qn @ d_block.T

            # apply doc mask for this block
            if (~dmask_block).any():
                sim_block[:, ~dmask_block] = self.neg

            # update argmax / best scores
            block_max = sim_block.max(axis=1).astype(np.float32)     # (Lq,)
            block_arg = sim_block.argmax(axis=1).astype(np.int32)    # (Lq,)

            better = block_max > best_scores
            if better.any():
                best_scores[better] = block_max[better]
                best_idx[better] = (start + block_arg[better]).astype(np.int32)

            # update per-token topk if needed
            if need_topk:
                B = end - start
                K = topk_scores.shape[1]
                k_block = min(K, B)

                # per row topk in this block
                blk_sel = np.argpartition(sim_block, -k_block, axis=1)[:, -k_block:]
                blk_vals = np.take_along_axis(sim_block, blk_sel, axis=1).astype(np.float32)
                blk_idx = (start + blk_sel).astype(np.int32)

                # sort within block topk desc
                blk_order = np.argsort(blk_vals, axis=1)[:, ::-1]
                blk_vals = np.take_along_axis(blk_vals, blk_order, axis=1)
                blk_idx = np.take_along_axis(blk_idx, blk_order, axis=1)

                # if k_block < K, pad
                if k_block < K:
                    pad_scores = np.full((Lq, K - k_block), self.neg, dtype=np.float32)
                    pad_idx = np.full((Lq, K - k_block), -1, dtype=np.int32)
                    blk_vals = np.concatenate([blk_vals, pad_scores], axis=1)
                    blk_idx = np.concatenate([blk_idx, pad_idx], axis=1)

                topk_scores, topk_idx = _merge_topk(topk_scores, topk_idx, blk_vals, blk_idx, K)

        # apply query mask: invalid query tokens contribute 0 and have idx -1
        best_scores_masked = best_scores.copy()
        best_scores_masked[~q_mask] = 0.0
        best_idx_masked = best_idx.copy()
        best_idx_masked[~q_mask] = -1

        score = float(best_scores_masked.sum())

        if artifacts_level == "none":
            return ScoreBreakdown(score=score)

        stats: Dict[str, Any] = {
            "Lq": int(Lq),
            "Ld": int(Ld),
            "valid_q": int(q_mask.sum()),
            "valid_d": int(d_mask.sum()),
            "block_size": int(self.block_size),
            "token_score_mean": float(best_scores_masked[q_mask].mean()) if q_mask.any() else 0.0,
            "token_score_std": float(best_scores_masked[q_mask].std()) if q_mask.any() else 0.0,
        }

        # chunk/page aggregation (optional)
        chunk_info = _chunk_aggregate(best_scores_masked, best_idx_masked, q_mask, d_meta)
        if chunk_info is not None:
            stats.update(chunk_info)

        if artifacts_level == "token":
            artifacts = InteractionArtifacts(
                token_scores=best_scores_masked,
                token_to_doc_idx=best_idx_masked,
                stats=stats,
            )
            return ScoreBreakdown(score=score, artifacts=artifacts)

        # artifacts_level == "topk"
        # mask out invalid query tokens
        if (~q_mask).any():
            topk_scores[~q_mask, :] = 0.0
            topk_idx[~q_mask, :] = -1

        artifacts = InteractionArtifacts(
            token_scores=best_scores_masked,
            token_to_doc_idx=best_idx_masked,
            topk_doc_idx=topk_idx,
            topk_scores=topk_scores,
            stats=stats,
        )
        return ScoreBreakdown(score=score, artifacts=artifacts)