import os
import json
import logging
import requests
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import List
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values


# -------------------------------
# Setup
# -------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

DIR = "../data/recepten/13k-recipes.csv"
APIKEYOPENROUTER = os.getenv("APIKEYOPENROUTER")
APIKEYQWEN = os.getenv("APIKEYQWEN")
DIMEMBEDDING = 768
BATCH_SIZE = 50

QWEN_URL = "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Embedding-8B"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-5-nano"

HEADERSOPENROUTER = {
    "Authorization": f"Bearer {APIKEYOPENROUTER}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

SYSTEM_PROMPT = (
    "Convert all ingredients and instructions in the following recipe from "
    "imperial units (including abbreviations such as tsp, tbsp, oz, lb, fl oz, °F, etc.) "
    "to metric units. Replace cups, ounces, pounds, teaspoons, tablespoons, and Fahrenheit "
    "with the appropriate metric units (grams, milliliters, Celsius). Recalculate the values "
    "accurately and update the recipe text, making sure the metric units are written out "
    "clearly (e.g., g for grams, ml for milliliters, °C for Celsius). Preserve the recipe's "
    "original structure and clarity. Make everything more readable so delete unnecessary "
    "characters. Remove unnecessary words in the ingredients like 'about', etc ."
)

HEADERSQWEN = {
    "Authorization": f"bearer {APIKEYQWEN}",
    "Content-Type": "application/json",
}


# -------------------------------
# Models
# -------------------------------
class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    instructions: str


# -------------------------------
# Functions
# -------------------------------
def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def call_openrouter(row) -> Recipe:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Title:{row.Title}, Ingredients:{row.Ingredients}, "
                    f"Instructions:{row.Instructions}, Cleaned_Ingredients:{row.Cleaned_Ingredients}"
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "recipe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "ingredients": {"type": "array", "items": {"type": "string"}},
                        "instructions": {"type": "string"},
                    },
                    "required": ["title", "ingredients", "instructions"],
                    "additionalProperties": False,
                },
            },
        },
    }

    response = requests.post(
        OPENROUTER_URL, headers=HEADERSOPENROUTER, json=payload, timeout=60
    )

    if not response.ok:
        logging.error("API Error %s: %s", response.status_code, response.text[:200])
        raise RuntimeError(f"OpenRouter error {response.status_code}")

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        recipe_dict = json.loads(content) if isinstance(content, str) else content
        return Recipe(**recipe_dict)
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        logging.error("Parsing error: %s", e)
        logging.debug("Raw response: %s", response.text[:500])
        raise


def get_embedding(
    text: str, dimensions: int = DIMEMBEDDING, normalize: bool = True
) -> list[float]:
    payload = {
        "inputs": [text],  # lijst van strings
        "normalize": normalize,
        "dimensions": dimensions,
    }

    response = requests.post(QWEN_URL, headers=HEADERSQWEN, json=payload, timeout=60)
    if not response.ok:
        raise RuntimeError(
            f"DeepInfra error {response.status_code}: {response.text[:300]}"
        )

    data = response.json()
    # Veelvoorkomende responsvormen afvangen
    if "embeddings" in data:
        emb = data["embeddings"][0]
    elif (
        "data" in data
        and isinstance(data["data"], list)
        and "embedding" in data["data"][0]
    ):
        emb = data["data"][0]["embedding"]
    else:
        raise ValueError(f"Onverwacht embedding response formaat: {list(data.keys())}")

    if not isinstance(emb, list) or not all(isinstance(x, (float, int)) for x in emb):
        raise TypeError("Embedding is geen lijst van numerieke waarden")

    return [float(x) for x in emb]


# -------------------------------
# Main
# -------------------------------
def main():
    df = pd.read_csv(DIR).drop(columns=["Index", "Image_Name"])

    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()

    rows_buffer = []  # elke entry: (document, embedding)

    try:
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            recipe = call_openrouter(row)

            document = (
                f"{recipe.title}\n\nIngredients:\n"
                + "\n".join(recipe.ingredients)
                + "\n\nInstructions:\n"
                + recipe.instructions
            )
            embedding = get_embedding(document, dimensions=DIMEMBEDDING)

            rows_buffer.append((document, embedding))

            if len(rows_buffer) == BATCH_SIZE:
                execute_values(
                    cur,
                    """
                    INSERT INTO embedding (document, embedding)
                    VALUES %s
                    """,
                    rows_buffer,
                    template="(%s, %s)",
                    page_size=BATCH_SIZE,
                )
                conn.commit()
                logging.info(
                    "💾 Batch van %d embeddings gecommit (t/m record %d)",
                    BATCH_SIZE,
                    idx,
                )
                rows_buffer.clear()

        # resterende (< BATCH_SIZE) nog wegschrijven
        if rows_buffer:
            execute_values(
                cur,
                """
                INSERT INTO embedding (document, embedding)
                VALUES %s
                """,
                rows_buffer,
                template="(%s, %s)",
                page_size=len(rows_buffer),
            )
            conn.commit()
            logging.info("✅ Laatste %d embeddings gecommit", len(rows_buffer))

    except Exception as e:
        conn.rollback()
        logging.error("❌ Fout tijdens verwerken/inserten: %s", e)
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
