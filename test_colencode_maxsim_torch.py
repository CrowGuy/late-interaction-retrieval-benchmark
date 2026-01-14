import requests
from io import BytesIO
from PIL import Image
import numpy as np

from src.config.encoders import EncoderConfig
from src.encoders.col_encoder import ColEncoder
from src.interactions.maxsim_torch import MaxSimTorch


def load_image(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def main():
    # 1) Init encoder (choose one)
    cfg = EncoderConfig(
        model_name="/home/randy/Documents/vlm/models/colpali-v1.3",  # change if needed
        model_kind="colpali",
        device="cuda",  # or "cpu"
        # if you use auto models, you can pass processor/model kwargs here
        # processor_kwargs={"max_num_visual_tokens": 1280},
        # model_kwargs={"attn_implementation": "flash_attention_2"},
    )
    enc = ColEncoder(cfg)

    # 2) Prepare inputs
    query = "carbon capture from gas processing in 2018"
    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/20210413_Carbon_capture_and_storage_-_CCS_-_proposed_vs_implemented.svg/2560px-20210413_Carbon_capture_and_storage_-_CCS_-_proposed_vs_implemented.svg.png"
    pil_image = load_image(img_url)

    # 3) Encode
    q_repr = enc.encode_query(text=query)
    d_repr = enc.encode_doc(image=pil_image, doc_id="demo_p1")

    print("== Encoder outputs ==")
    print("q emb shape:", q_repr["emb"].shape, "dtype:", q_repr["emb"].dtype)
    print("d emb shape:", d_repr["emb"].shape, "dtype:", d_repr["emb"].dtype)
    print("q meta keys:", list(q_repr["meta"].keys()))
    print("d meta keys:", list(d_repr["meta"].keys()))

    # sanity: mask existence (not all models provide all masks)
    if "attention_mask" in q_repr["meta"]:
        print("q attention_mask shape:", q_repr["meta"]["attention_mask"].shape)
    if "image_mask" in d_repr["meta"]:
        print("d image_mask shape:", d_repr["meta"]["image_mask"].shape)
    elif "attention_mask" in d_repr["meta"]:
        print("d attention_mask shape:", d_repr["meta"]["attention_mask"].shape)

    # 4) Interaction: MaxSim masked
    inter = MaxSimTorch(device="cuda", block_size=2048, assume_normalized=False)

    # token-level artifacts
    out_token = inter.score(q_repr, d_repr, artifacts_level="token")
    print("\n== MaxSim token artifacts ==")
    print("score:", out_token.score)
    print("token_scores shape:", out_token.artifacts.token_scores.shape)
    print("token_to_doc_idx shape:", out_token.artifacts.token_to_doc_idx.shape)
    print("stats keys:", list(out_token.artifacts.stats.keys()))

    # topk artifacts
    out_topk = inter.score(q_repr, d_repr, artifacts_level="topk", topk=5)
    print("\n== MaxSim topk artifacts ==")
    print("score:", out_topk.score)
    print("topk_doc_idx shape:", out_topk.artifacts.topk_doc_idx.shape)
    print("topk_scores shape:", out_topk.artifacts.topk_scores.shape)

    # 5) Optional: chunk aggregation test
    # We'll fake "chunks" as 4 equal ranges across doc token axis.
    Ld = d_repr["emb"].shape[0]
    step = max(1, Ld // 4)
    chunks = []
    start = 0
    cid = 0
    while start < Ld:
        end = min(Ld, start + step)
        chunks.append({"chunk_id": cid, "start": start, "end": end})
        start = end
        cid += 1

    d_repr2 = {"emb": d_repr["emb"], "meta": dict(d_repr["meta"])}
    d_repr2["meta"]["chunks"] = chunks

    out_chunk = inter.score(q_repr, d_repr2, artifacts_level="token")
    print("\n== Chunk aggregation (synthetic) ==")
    stats = out_chunk.artifacts.stats
    if "chunk_scores" in stats:
        print("top_chunks:", stats["top_chunks"][:3])
    else:
        print("No chunk_scores found. (Did you run the chunk-agg version of maxsim.py?)")

    print("\n== Verify masking ==")
    d_repr_all_masked = {"emb": d_repr["emb"], "meta": dict(d_repr["meta"])}
    d_repr_all_masked["meta"]["image_mask"] = np.zeros_like(d_repr["meta"]["image_mask"], dtype=bool)
    out = inter.score(q_repr, d_repr_all_masked, artifacts_level="token")
    print("score with all-false image_mask:", out.score)

    print("\n== Verify the padding mask of query side")
    q_repr_masked = {"emb": q_repr["emb"], "meta": dict(q_repr["meta"])}
    q_mask = np.ones((q_repr["emb"].shape[0],), dtype=bool)
    q_mask[-5:] = False  # 假裝最後 5 個 token 是 padding
    q_repr_masked["meta"]["attention_mask"] = q_mask
    out_full = inter.score(q_repr, d_repr, artifacts_level="token")
    out_mask = inter.score(q_repr_masked, d_repr, artifacts_level="token")
    print("full score:", out_full.score)
    print("masked score:", out_mask.score)

if __name__ == "__main__":
    main()