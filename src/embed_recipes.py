# -*- coding: utf-8 -*-
"""
Embed recepten met Hugging Face (intfloat/e5-base-v2), zonder LM Studio of trust_remote_code.
"""

import os
import json
import logging
from typing import List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------------
# Setup
# -------------------------------
load_dotenv("../configs/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

PATH_TRANSFORM_DRECIPES = "../data/transformed_recipes.ndjson"

MODEL_ID   = "intfloat/e5-base-v2"
DEVICE_SEL = os.getenv("DEVICE", "auto").lower()  # auto|cpu|cuda
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))


# -------------------------------
# Postgres
# -------------------------------
def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


# -------------------------------
# Device & model
# -------------------------------
def detect_device(sel: str = "auto") -> torch.device:
    if sel == "cpu":
        return torch.device("cpu")
    if sel in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(x.norm(p=2, dim=-1, keepdim=True), min=eps)


class HFEmbedder:
    def __init__(self, model_id: str, device: torch.device, max_length: int = 512):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length
        self.emb_dim = getattr(self.model.config, "hidden_size", None)

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        # E5: prefix "passage: " voor documenten
        texts = [f"passage: {t}" for t in texts]

        all_embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                token_embeddings = outputs.last_hidden_state
                pooled = mean_pool(token_embeddings, enc["attention_mask"])
                if normalize:
                    pooled = l2_normalize(pooled)
                vecs = pooled.detach().cpu().float().numpy()
                all_embeddings.append(vecs)

        arr = np.concatenate(all_embeddings, axis=0)
        if self.emb_dim is None and arr.ndim == 2:
            self.emb_dim = arr.shape[1]
        return arr


# -------------------------------
# Data helpers
# -------------------------------
def build_document(title: str, ingredients: List[str], instructions: str) -> str:
    return (
        f"{title}\n\nIngredients:\n" + "\n".join(ingredients) + "\n\nInstructions:\n" + instructions
    )


def load_recipes_from_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            title = str(obj.get("title", ""))
            ingredients = obj.get("ingredients", [])
            if not isinstance(ingredients, list):
                ingredients = [str(ingredients)]
            ingredients = [str(x) for x in ingredients]
            instructions = str(obj.get("instructions", ""))
            yield {"title": title, "ingredients": ingredients, "instructions": instructions}


# -------------------------------
# Main
# -------------------------------
def main():
    device = detect_device(DEVICE_SEL)
    logging.info("Model: %s | Device: %s", MODEL_ID, device)

    embedder = HFEmbedder(MODEL_ID, device=device, max_length=MAX_LENGTH)

    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()

    total = 0
    buffer: List[Tuple[str, List[float]]] = []
    docs: List[str] = []

    try:
        for idx, rec in enumerate(load_recipes_from_jsonl(PATH_TRANSFORM_DRECIPES), start=1):
            doc = build_document(rec["title"], rec["ingredients"], rec["instructions"])
            docs.append(doc)

            if len(docs) == BATCH_SIZE:
                embs = embedder.encode(docs, batch_size=BATCH_SIZE)
                rows = [(doc, emb.tolist()) for doc, emb in zip(docs, embs)]
                execute_values(cur, "INSERT INTO embedding (document, embedding) VALUES %s", rows)
                conn.commit()
                total += len(docs)
                logging.info("💾 Batch %d gecommit, totaal %d", len(docs), total)
                docs.clear()

        if docs:
            embs = embedder.encode(docs, batch_size=len(docs))
            rows = [(doc, emb.tolist()) for doc, emb in zip(docs, embs)]
            execute_values(cur, "INSERT INTO embedding (document, embedding) VALUES %s", rows)
            conn.commit()
            total += len(docs)
            logging.info("✅ Laatste batch gecommit (%d). Totaal: %d", len(docs), total)

    except Exception as e:
        conn.rollback()
        logging.error("❌ Fout: %s", e)
        raise
    finally:
        cur.close()
        conn.close()

    logging.info("Klaar. Totaal verwerkte rijen: %d", total)


if __name__ == "__main__":
    main()