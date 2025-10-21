# embedder.py
"""
Text and image embedding helpers.

Requirements:
 - transformers
 - sentence-transformers
 - torch
 - requests
 - pillow
"""

from io import BytesIO
import requests
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

# Text model (sentence-transformers)
_TEXT_MODEL_NAME = "all-mpnet-base-v2"  # change if you prefer another
text_model = SentenceTransformer(_TEXT_MODEL_NAME)

# CLIP model for images (Hugging Face transformer variant)
_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """
    Returns a normalized embedding for the input text (list of floats).
    """
    if text is None:
        text = ""
    # SentenceTransformer encode with normalization recommended
    emb = text_model.encode(text, normalize_embeddings=True)
    return emb.tolist()


def embed_image_from_url(url: str, timeout: int = 12) -> list[float]:
    """
    Download image from URL and return CLIP image embedding (normalized).
    Raises on network/processing errors.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    emb = outputs.detach().cpu().numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.tolist()


def get_text_embedding_dimension() -> int:
    """
    Return dimension of text embeddings.
    """
    sample = embed_text("test")
    return len(sample)


def get_image_embedding_dimension() -> int:
    """
    Return dimension of image embeddings (CLIP).
    """
    # Use a tiny black image to get dims
    dummy = Image.new("RGB", (224, 224), (0, 0, 0))
    inputs = clip_processor(images=dummy, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs.detach().cpu().numpy()[0].shape[-1]
