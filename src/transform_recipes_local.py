# -*- coding: utf-8 -*-
"""
Lokaal recepten transformeren via LM Studio (openai/gpt-oss-20b)
- Converteert imperial → metrisch (g/ml/°C), schoont tekst op
- Produceert strikt JSON (title, ingredients[], instructions)
- Schrijft NDJSON (1 recept per regel)
- Draait tegen LM Studio's OpenAI-compatibele endpoint
"""

import os
import json
import logging
from typing import List, Optional

import requests
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator

# -------------------------------
# Setup
# -------------------------------
load_dotenv("../configs/.env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

DIR = os.getenv("INPUT_CSV", "../data/13k-recipes.csv")
OUTPUT_FILE = "../data/transformed_recipes2.ndjson"
ERROR_FILE = "../data/failed_recipes2.txt"
TEST_MODE = False  # True = alleen eerste recept

# LM Studio / OpenAI-compatibele API
LM_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-20b")

# Generatieparameters
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SECONDS", "120"))

# -------------------------------
# Prompt (strikt JSON)
# -------------------------------
ALWAYS_JSON_INSTRUCTIONS = (
    "OUTPUT FORMAT RULES (STRICT):\n"
    "1) Return EXACTLY ONE valid JSON object. No markdown, no code fences, no explanations.\n"
    "2) Keys and value types:\n"
    "   - \"title\": string (non-empty)\n"
    "   - \"ingredients\": array of strings (each item = a single ingredient line; concise; metric units)\n"
    "   - \"instructions\": string (concise, numbered or clearly delimited steps in one string)\n"
    "3) JSON hygiene: double quotes for all strings; no trailing commas; UTF-8 only.\n"
    "4) Do not include any extra keys or metadata.\n"
    "5) Use METRIC units ONLY: grams (g) for solids, milliliters (ml) for liquids, Celsius (°C) for temperature.\n"
    "6) Be precise and consistent. Remove vague words (about/approximately/~/±). Convert ranges to a single precise value (use midpoint) unless a range is essential to safety.\n"
    "7) Keep ingredient names intact (brands optional), normalize amounts and units, and consolidate duplicates (sum same ingredients).\n"
    "8) Don’t invent ingredients or steps. If a quantity truly cannot be inferred, keep the item but omit guessing; write a clean, unitless line.\n"
    "9) Prefer WEIGHT for baking-critical items. Normalize fractions to decimals (e.g., 1/2 → 0.5).\n"
    "10) Temperatures in °C; times in minutes/hours with integers (e.g., 25 min, 1 hr 30 min).\n"
    "11) Spelling/formatting: use lowercase units with a space (e.g., \"200 g\", \"150 ml\"); pluralize ingredient names naturally.\n"
    "JSON EXAMPLE (structure only):\n"
    "{\"title\":\"…\",\"ingredients\":[\"200 g flour\",\"150 ml milk\",\"2 eggs\"],\"instructions\":\"1. … 2. … 3. …\"}\n"
)

SYSTEM_PROMPT = (
    (
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
    + ALWAYS_JSON_INSTRUCTIONS
)

# -------------------------------
# Datamodel
# -------------------------------
class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    instructions: str

    @field_validator("title", "instructions", mode="before")
    @classmethod
    def strip_text(cls, v):
        return (v or "").strip()

    @field_validator("ingredients", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v]
        return [str(v).strip()]

# -------------------------------
# HTTP helpers
# -------------------------------
def http_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LM_API_KEY}",
    }

def explain_http_error(resp: requests.Response) -> str:
    try:
        data = resp.json()
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return (resp.text or "")[:4000]

def ensure_model_available(model_id: str):
    try:
        r = requests.get(f"{LM_BASE_URL}/models", headers=http_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        models = r.json().get("data", [])
        ids = [m.get("id") for m in models if isinstance(m, dict)]
        if model_id not in ids:
            logging.error("Model '%s' niet gevonden in LM Studio. Beschikbare ids: %s", model_id, ids)
            raise RuntimeError(
                f"Model '{model_id}' niet geladen. Start/Selecteer het model in LM Studio. "
                f"Beschikbare ids (LM Studio): {ids}"
            )
        logging.info("Model '%s' is beschikbaar in LM Studio.", model_id)
    except requests.HTTPError as e:
        msg = explain_http_error(e.response)
        raise RuntimeError(f"Kon /v1/models niet ophalen ({e}). Server zei: {msg}") from e

# -------------------------------
# Prompt helpers
# -------------------------------
def build_messages(system: str, title: str, ingredients: str, instructions: str, cleaned_ingredients: Optional[str]):
    user = (
        "TASK:\n"
        "Convert the following recipe to metric units and return ONLY the JSON described in the rules.\n\n"
        f"Title: {title}\n"
        f"Ingredients: {ingredients}\n"
        f"Instructions: {instructions}\n"
    )
    if cleaned_ingredients:
        user += f"Cleaned_Ingredients (optional preprocessed list): {cleaned_ingredients}\n"
    user += "\nREMINDER: Return exactly one JSON object and nothing else."
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]

def build_repair_messages(bad_text: str):
    """Tweede poging: forceer strikt JSON op basis van mislukte output."""
    system = (
        "You are a JSON fixer. You receive possibly invalid or non-JSON content and MUST output "
        "a single valid JSON object that satisfies the schema and rules below. Do NOT explain."
        "\n\n" + ALWAYS_JSON_INSTRUCTIONS
    )
    user = (
        "Fix the following content into a single valid JSON object per the schema. "
        "Return only the JSON object:\n\n"
        f"{bad_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# -------------------------------
# LLM calls
# -------------------------------
def call_chat_completions(messages: list) -> str:
    url = f"{LM_BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": False,
    }
    resp = requests.post(url, headers=http_headers(), json=payload, timeout=TIMEOUT)
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {url}\n{explain_http_error(resp)}", response=resp)
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def call_completions_fallback(messages: list) -> str:
    url = f"{LM_BASE_URL}/completions"
    sys = next((m["content"] for m in messages if m["role"] == "system"), "")
    usr = next((m["content"] for m in messages if m["role"] == "user"), "")
    prompt = f"{sys}\n\n{usr}\n\nRespond ONLY with valid JSON, no extra text."
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": False,
        "stop": None,
    }
    resp = requests.post(url, headers=http_headers(), json=payload, timeout=TIMEOUT)
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {url}\n{explain_http_error(resp)}", response=resp)
    data = resp.json()
    return data["choices"][0]["text"]

def generate_with_retry(messages: list, use_fallback_on_error=True) -> str:
    """
    1e poging: /chat/completions → JSON parse
    Als parsing faalt: 2e poging met JSON-fixer.
    Als chat endpoint faalt: probeer /completions; daarna evt. fixer.
    Retourneert ALTIJD de laatst verkregen ruwe text (kan nog steeds invalid zijn).
    """
    raw = None

    def try_parse(text: str) -> Optional[dict]:
        try:
            return extract_first_json(text)
        except Exception:
            return None

    # --- eerste poging ---
    try:
        raw = call_chat_completions(messages)
        if try_parse(raw) is not None:
            return raw
        # tweede poging: repair via fixer
        repair_raw = call_chat_completions(build_repair_messages(raw))
        return repair_raw
    except requests.HTTPError:
        if not use_fallback_on_error:
            raise
        # fallback naar completions
        try:
            raw = call_completions_fallback(messages)
            if try_parse(raw) is not None:
                return raw
            repair_raw = call_completions_fallback(build_repair_messages(raw))
            return repair_raw
        except Exception:
            # geef door aan bovenliggende except
            raise

# -------------------------------
# JSON helpers
# -------------------------------
def extract_first_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("Geen '{' gevonden in modeloutput")
    depth = 0
    candidate_chars = []
    for ch in text[start:]:
        candidate_chars.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = "".join(candidate_chars)
                return json.loads(candidate)
    raise ValueError("Kon geen gesloten JSON-object extraheren uit modeloutput")

def append_ndjson(obj: dict, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_error_line(index: int, raw_text: Optional[str]):
    """
    Schrijf index + ruwe output of server body. Newlines -> '\n' voor makkelijke parsing.
    """
    safe = "" if raw_text is None else str(raw_text).replace("\n", "\\n")
    with open(ERROR_FILE, "a", encoding="utf-8") as ef:
        ef.write(f"{index}\t{safe}\n")

# -------------------------------
# Main
# -------------------------------
def main():
    logging.info("LM Studio endpoint: %s | Model: %s", LM_BASE_URL, MODEL_ID)
    ensure_model_available(MODEL_ID)

    df = pd.read_csv(DIR)
    for col in ["Index", "Image_Name"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Zorg dat outputbestanden bestaan/leeg zijn
    open(OUTPUT_FILE, "a", encoding="utf-8").close()
    open(ERROR_FILE, "a", encoding="utf-8").close()

    processed = 0
    for i, row in enumerate(df.itertuples(index=False), start=0):
        raw = None
        try:
            title = str(getattr(row, "Title", "") or "")
            ingredients = str(getattr(row, "Ingredients", "") or "")
            instructions = str(getattr(row, "Instructions", "") or "")
            cleaned = getattr(row, "Cleaned_Ingredients", None)
            cleaned = str(cleaned) if cleaned is not None and str(cleaned).strip() else None

            def soft_trunc(s: str, n: int = 16000) -> str:
                return s if s is None or len(s) <= n else s[:n]

            messages = build_messages(
                SYSTEM_PROMPT,
                soft_trunc(title, 512),
                soft_trunc(ingredients, 12000),
                soft_trunc(instructions, 16000),
                soft_trunc(cleaned, 12000) if cleaned else None,
            )

            # Genereer (met auto-repair) en parse
            raw = generate_with_retry(messages)
            recipe_dict = extract_first_json(raw)  # parse moet nu slagen
            recipe = Recipe(**recipe_dict)

            append_ndjson(recipe.model_dump(), OUTPUT_FILE)
            logging.info("✅ Recept %d getransformeerd en weggeschreven", i)
            processed += 1

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logging.error("❌ JSON/validatie-fout bij recept %d: %s", i, e)
            write_error_line(i, raw)
        except requests.HTTPError as e:
            logging.error("❌ HTTP-fout bij recept %d: %s", i, e)
            body = explain_http_error(e.response) if getattr(e, "response", None) else str(e)
            write_error_line(i, body)
        except Exception as e:
            logging.error("❌ Onverwachte fout bij recept %d: %s", i, e)
            write_error_line(i, raw if raw else str(e))

        if TEST_MODE:
            break

    logging.info("💾 Klaar. %d recept(en) opgeslagen in %s", processed, OUTPUT_FILE)

if __name__ == "__main__":
    main()