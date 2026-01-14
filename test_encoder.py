from PIL import Image
from io import BytesIO
import requests

from src.config.encoders import EncoderConfig
from src.encoders.col_encoder import ColEncoder

def main():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    pil_image = Image.open(BytesIO(requests.get("https://upload.wikimedia.org/wikipedia/commons/2/27/Singapore_skyline_2022.jpg", headers=headers).content)).convert("RGB")
    """
    cfg = EncoderConfig(
        model_name="/home/randy/Documents/vlm/models/colpali-v1.3",
        model_kind="colpali",
        device="cuda",  # or "cpu"
    )
    enc = ColEncoder(cfg)

    q = enc.encode_query(text="what is carbon capture?")
    print("Q emb:", q["emb"].shape, q["emb"].dtype)
    
    d = enc.encode_doc(image=pil_image, doc_id="paper1_p3")
    print("D emb:", d["emb"].shape, d["emb"].dtype)
    print("D meta keys:", list(d["meta"].keys()))

    """
    cfg = EncoderConfig(
        model_name="/home/randy/Documents/vlm/models/tomoro-colqwen3-embed-8b",
        model_kind="auto",
        processor_kwargs={"max_num_visual_tokens": 1280},
    )
    enc = ColEncoder(cfg)
    q = enc.encode_query(text="what is carbon capture?")
    print("Q emb:", q["emb"].shape, q["emb"].dtype)

    d = enc.encode_doc(image=pil_image, doc_id="paper1_p3")
    print("D emb:", d["emb"].shape, d["emb"].dtype)
    print("D meta keys:", list(d["meta"].keys()))

if __name__ == "__main__":
    main()



