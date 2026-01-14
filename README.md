```text
src/
  encoders/
    __init__.py
    base.py
    colpali_encoder.py
    loader.py

  interactions/
    base.py
    maxsim.py
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