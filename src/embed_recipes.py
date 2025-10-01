import os
import json
import logging
import requests
import psycopg2
from dotenv import load_dotenv
from typing import List
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

# -------------------------------
# Setup
# -------------------------------
load_dotenv("../configs/.env")

PATH_TRANSFORM_DRECIPES = "../data/transformed_recipes.ndjson"

# Postgres
def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

# Embedding instellingen
DIMEMBEDDING = 1024   # standaarddim van text-embedding-qwen3-embedding-0.6b, 2560 4B, 4096 8B
BATCH_SIZE = 50

# --- LM Studio (lokaal) ---
LM_STUDIO_BASE = os.getenv("LM_STUDIO_BASE", "http://localhost:1234/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b")
LM_HEADERS = {"Content-Type": "application/json"}

# --- (commentaar) Remote API (DeepInfra) ---
# QWEN_URL = "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Embedding-8B"
# APIKEYQWEN = os.getenv("APIKEYQWEN")
# HEADERSQWEN = {
#     "Authorization": f"bearer {APIKEYQWEN}",
#     "Content-Type": "application/json",
# }

# -------------------------------
# Embedding
# -------------------------------
def get_embedding(text: str) -> list[float]:
    # --- Lokaal (LM Studio) ---
    payload_local = payload_local = {"model": EMBED_MODEL, "input": text, "dimensions": 768}
    r = requests.post(f"{LM_STUDIO_BASE}/embeddings", headers=LM_HEADERS, json=payload_local, timeout=60)
    if not r.ok:
        raise RuntimeError(f"LM Studio error {r.status_code}: {r.text[:300]}")
    data = r.json()
    if "data" not in data or not data["data"] or "embedding" not in data["data"][0]:
        raise ValueError(f"Onverwacht LM Studio response: {data}")
    emb = data["data"][0]["embedding"]
    return [float(x) for x in emb]

    # --- (COMMENTAAR) Remote API via DeepInfra (Qwen3-Embedding-8B) ---
    # payload_remote = {
    #     "inputs": [text],
    #     "normalize": True,
    #     "dimensions": DIMEMBEDDING,
    # }
    # r = requests.post(QWEN_URL, headers=HEADERSQWEN, json=payload_remote, timeout=60)
    # if not r.ok:
    #     raise RuntimeError(f"DeepInfra error {r.status_code}: {r.text[:300]}")
    # data = r.json()
    # if "embeddings" in data:
    #     emb = data["embeddings"][0]
    # elif "data" in data and isinstance(data["data"], list) and "embedding" in data["data"][0]:
    #     emb = data["data"][0]["embedding"]
    # else:
    #     raise ValueError(f"Onverwacht embedding response formaat: {list(data.keys())}")
    # return [float(x) for x in emb]

# -------------------------------
# Helpers
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
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()

    rows_buffer = []
    count = 0

    try:
        for idx, rec in enumerate(load_recipes_from_jsonl(PATH_TRANSFORM_DRECIPES ), start=1):
            doc = build_document(rec["title"], rec["ingredients"], rec["instructions"])
            emb = get_embedding(doc)
            if len(emb) != DIMEMBEDDING:
                raise ValueError(f"Embedding-dim mismatch: verwacht {DIMEMBEDDING}, kreeg {len(emb)}")
            rows_buffer.append((doc, emb))
            count += 1
            logging.info("➡️ Rij %d verwerkt", idx)

            if len(rows_buffer) == BATCH_SIZE:
                execute_values(cur, "INSERT INTO embedding (document, embedding) VALUES %s", rows_buffer)
                conn.commit()
                logging.info("💾 Batch van %d embeddings gecommit (t/m record %d)", BATCH_SIZE, idx)
                rows_buffer.clear()

        if rows_buffer:
            execute_values(cur, "INSERT INTO embedding (document, embedding) VALUES %s", rows_buffer)
            conn.commit()
            logging.info("✅ Laatste %d embeddings gecommit", len(rows_buffer))

        logging.info("Klaar. Totaal verwerkte rijen: %d", count)

    except Exception as e:
        conn.rollback()
        logging.error("❌ Fout tijdens embedden/inserten: %s", e)
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
