import os
import json
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import List

# -------------------------------
# Setup
# -------------------------------
load_dotenv("../configs/.env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

DIR = "../data/recepten/13k-recipes.csv"
APIKEYOPENROUTER = os.getenv("APIKEYOPENROUTER")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5-nano"

SYSTEM_PROMPT = (
    "Convert all ingredients and instructions in the following recipe from "
    "imperial units (including abbreviations such as tsp, tbsp, oz, lb, fl oz, °F, etc.) "
    "to metric units. Replace cups, ounces, pounds, teaspoons, tablespoons, and Fahrenheit "
    "with the appropriate metric units (grams, milliliters, Celsius). Recalculate the values "
    "accurately and update the recipe text, making sure the metric units are written out "
    "clearly (e.g., g for grams, ml for milliliters, °C for Celsius). Preserve the recipe's "
    "original structure and clarity. Make everything more readable so delete unnecessary "
    "characters. Remove unnecessary words in the ingredients like 'about', etc. The output "
    "must be in valid JSON. "
)

HEADERSOPENROUTER = {
    "Authorization": f"Bearer {APIKEYOPENROUTER}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

OUTPUT_FILE = "./transformed_recipes.ndjson"  # NDJSON (1 JSON-object per regel)
ERROR_FILE = "./failed_recipes.txt"
TEST_MODE = False   # Zet True voor 1 recept, False voor alles

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

    response = requests.post(OPENROUTER_URL, headers=HEADERSOPENROUTER, json=payload, timeout=60)
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

def append_ndjson(obj: dict, path: str):
    """Schrijf één JSON-object als nieuwe regel naar een NDJSON-bestand."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------------------------------
# Main
# -------------------------------
def main():
    df = pd.read_csv(DIR).drop(columns=["Index", "Image_Name"])

    for i, row in enumerate(df.itertuples(index=False), start=0):
        try:
            recipe = call_openrouter(row)
            append_ndjson(recipe.model_dump(), OUTPUT_FILE)
            logging.info("✅ Recept %d getransformeerd en weggeschreven", i)
        except Exception as e:
            logging.error("❌ Recept %d mislukt: %s", i, e)
            with open("ERROR_FILE", "a", encoding="utf-8") as f:
                f.write(f"Recept {i}\n")

        if TEST_MODE:
            break  # alleen eerste recept verwerken

    logging.info("💾 Klaar. Resultaten in %s (NDJSON)", OUTPUT_FILE)

if __name__ == "__main__":
    main()