# Late-Interaction Retrieval Benchmark (WIP)

A late-interaction-aware retrieval evaluation framework focusing on:
- Model-agnostic encoder interface (query/doc representations)
- Interaction-aware scoring with rich artifacts (token-level alignment, score distribution)
- Optional native scoring adapters for sanity checks
- (Next) Index + dataset adapters + pairwise/relative evaluation

## Project Structure

```text
src/
  config/
    encoders.py     # EncoderConfig + model-kind inference

  encoders/
    base.py         # (optional) encoder interface contract
    col_encoder.py  # unified encoder for ColPali / ColQwen / HF / Nemo-style models
    loader.py       # load model + processor (registry-based)

  interactions/
    base.py         # LateInteraction + ScoreBreakdown + InteractionArtifacts
    maxsim.py       # MaxSim (CPU / block-wise) + masking + (optional) chunk/page aggregation
    maxsim_torch.py # MaxSim (Torch/GPU / block-wise) + masking + valid_q-only topk optimization
    native.py       # NativeScoreInteraction (processor.score / model.get_score(s) / score_multi_vector)
    meansim.py
    weighted_max.py
    cross_patch.py   # experimental

  index/
    base.py
    faiss_flat.py
    faiss_ivf.py
    torch_index.py
    byaldi_compat.py

  datasets/
    base.py
    beir.py
    scidocs.py

  eval/
    base.py
    pointwise.py      # recall@k / nDCG@k / mrr@k
    pairwise.py       # pairwise win-rate / relative metrics
    analysis.py       # failure case mining + visualization-friendly artifacts

  pipeline.py         # 把 encoder/interaction/index 串起來（核心 runtime）
  types.py            # 共用型別：QueryRepr/DocRepr/InteractionArtifacts/ScoreBreakdown
```
---

## Core Concepts

### QueryRepr / DocRepr

Encoders output a representation dict:

```python
{
  "emb": np.ndarray of shape (L, D), dtype float32,
  "meta": dict
}
```
- `L` = number of token vectors (text tokens or image patch tokens)
- `D` = embedding dimension

### Interaction Artifacts (for failure analysis)

`LateInteraction.score(...)` returns:
```python
ScoreBreakdown(
  score: float,
  artifacts: InteractionArtifacts | None,
  debug: dict | None
)
```
Artifacts may include:
- `token_scores`: (Lq,) per query token contribution
- `token_to_doc_idx`: (Lq,) argmax aligned doc token index
- `topk_doc_idx/topk_scores`: (Lq, K) top-k alignments per query token
- `stats`: misc diagnostics (valid_q/valid_d, block_size, etc.)
- optional aggregation (if `d_meta["chunks"]` provided)
---
## Implemented Features (Current)
### 1) Encoder: `ColEncoder`

File: `src/encoders/col_encoder.py`

Supports multiple backends:
- ColPali / ColQwen (colpali-engine processors)
- Tomoro-style processors (`process_texts` / `process_images`)
- NemoRetriever-style models (no processor, `forward_queries` / `forward_images`)
- HF fallback via AutoProcessor/AutoModel (depends on model)

Masking support:
- Query: `meta["attention_mask"]` (if available)
- Doc: `meta["image_mask"]` (preferred, if available) else `meta["attention_mask"]`

### 2) Interaction: MaxSim (late interaction)
#### CPU block-wise
File: `src/interactions/maxsim.py`
- Does NOT materialize full `(Lq, Ld)` sim matrix
- Supports masking via query/doc masks
- Optional chunk/page aggregation if `d_meta["chunks"]` exists

#### Torch/GPU block-wise
File: `src/interactions/maxsim_torch.py`
- GPU accelerated block-wise matmul
- Supports masking
- `artifacts_level="topk"`: computes topk only for valid query tokens (based on query mask)

### 3) Interaction: Native scoring adapter
File: `src/interactions/native.py`

For sanity checks / parity with official demos:
- `processor.score(q_emb, d_emb)` (granite-like)
- `model.get_score / model.get_scores` (nemo-like)
- `processor.score_multi_vector([q],[d])` (colqwen-like demos)

Not guaranteed for every model; typically does NOT provide token-level artifacts.

---
## Quickstart
### 1) Install deps
(Your environment should include torch, transformers, colpali-engine, pillow, requests)

### 2) Run CPU MaxSim test
```bash
PYTHONPATH=src python test_colencode_maxsim.py
```

Expected:
- encoder outputs `q emb shape: (Lq, D)`, `d emb shape: (Ld, D)`
- MaxSim returns scalar score + artifacts

### 3) Run GPU MaxSim test
```bash
PYTHONPATH=src python test_colencode_maxsim_torch.py
```

Expected:
- similar score to CPU version (small numeric differences possible with mixed precision)
- artifacts include backend/device info in stats

---
## Notes on Chunk/Page Aggregation
`MaxSim` can optionally aggregate token contributions into chunk/page buckets if doc meta provides:
```python
d_repr["meta"]["chunks"] = [
  {"chunk_id": "page_1", "start": 0,   "end": 512},
  {"chunk_id": "page_2", "start": 512, "end": 1024},
]
```
Aggregation does NOT change the overall score.
It only explains "where the score came from" by summing query-token contributions into each chunk (page/section/region).

---
## Next Steps (Planned)
- Index layer:
  - ExactIndex (brute-force) for correctness
  - ANN coarse retrieval + rerank (MaxSim)

- Dataset adapters:
  - BEIRAdapter / SciDocsAdapter

- Evaluation:
  - pointwise metrics (MRR/nDCG/Recall@k)
  - pairwise/relative evaluation + failure breakdown reports
---
