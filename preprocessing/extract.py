# Installeer het package met: pip install llama-cloud (bij voorkeur in een aparte virtuele omgeving)

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import LlamaCloud

# Laad de omgevingsvariabelen in vanuit het .env bestand
load_dotenv(dotenv_path="../.env", override=True)

# Initialiseer de LlamaCloud client
client = LlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

# Definieer de mappen voor invoer en uitvoer
INPUT_DIR = "../data/rec_md"
OUTPUT_DIR = "../rec_extracted"

# Maak de outputmap aan als deze nog niet bestaat
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Schema voor gestructureerde data-extractie, afkomstig uit de playground
data_schema = {
    "description": "Structured extraction schema for recipe documents.",
    "type": "object",
    "properties": {
        "title": {
            "description": "The name of the recipe, in english",
            "type": "string"
        },
        "description": {
            "description": "A short introductory or summary description of the recipe, in english",
            "type": "string"
        },
        "category": {
            "description": "The primary category or course of the recipe, such as appetizer, main course, dessert, or snack, in english",
            "type": "string"
        },
        "tags": {
            "description": "Structured classification tags for the recipe, all in english.",
            "type": "object",
            "properties": {
                "meal_type": {
                    "description": "The type of meal this recipe is best suited for. Must be exactly one of the allowed values.",
                    "type": "string",
                    "enum": ["breakfast", "lunch", "dinner", "snack"]
                },
                "cuisine": {
                    "description": "The style of cuisine (e.g., Italian, Mexican, Thai). Use 'Global' if it does not belong to a specific cuisine.",
                    "type": "string"
                },
                "main_ingredients": {
                    "description": "Exactly the 3 most important main ingredients of the recipe, in english.",
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 3,
                    "maxItems": 3
                },
                "additional_tags": {
                    "description": "Any other useful tags (e.g., vegan, gluten-free, high-protein, quick, spicy), in english. Maximum of 5 tags.",
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["meal_type", "cuisine", "main_ingredients"],
            "additionalProperties": False
        },
        "servings": {
            "description": "The number of servings or people the recipe is intended for.",
            "type": "integer"
        },
        "prep_time": {
            "description": "The preparation time stated in the document, kept as written (for example, '15 min'), in english",
            "type": "string"
        },
        "difficulty": {
            "description": "The difficulty level of the recipe as stated in the document, in english",
            "type": "string"
        },
        "ingredients": {
            "description": "The list of ingredients required for the recipe, in english",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "description": "The ingredient name, in english",
                        "type": "string"
                    },
                    "quantity": {
                        "description": "The ingredient quantity, kept as written when provided, in english",
                        "type": "string"
                    },
                    "unit": {
                        "description": "The ingredient unit, if provided, in english",
                        "type": "string"
                    },
                    "preparation": {
                        "description": "Any preparation note associated with the ingredient, such as chopped or juiced, in english",
                        "type": "string"
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            }
        },
        "instructions": {
            "description": "Step-by-step preparation instructions.",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {
                        "description": "The step number in the preparation sequence.",
                        "type": "integer"
                    },
                    "text": {
                        "description": "The full text of the instruction step.",
                        "type": "string"
                    }
                },
                "required": ["step_number", "text"],
                "additionalProperties": False
            }
        }
    },
    "required": ["title", "tags", "ingredients", "instructions"],
    "additionalProperties": False
}

# Verwerk alle markdown bestanden in de invoermap
for file_path in Path(INPUT_DIR).glob("*.md"):
    
    # Upload het bestand naar LlamaCloud
    file_obj = client.files.create(file=file_path, purpose="extract")

    # Start een nieuwe extractie-job
    job = client.extract.create(
        file_input=file_obj.id,
        configuration={
            "data_schema": data_schema,
            "tier": "agentic",
            "extraction_target": "per_doc",
            "parse_tier": "agentic",
            "cite_sources": True,
            "confidence_scores": True
        },
    )

    # Blijf controleren (polling) totdat de job een eindstatus bereikt
    while job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
        time.sleep(2)
        job = client.extract.get(job.id)

    # Werp een foutmelding op als de job niet succesvol is afgerond
    if job.status != "COMPLETED":
        raise RuntimeError(f"Extract job {job.id} ended in {job.status}: {job.error_message}")

    # Sla het resulterende JSON-bestand lokaal op
    output_json = Path(OUTPUT_DIR) / f"{file_path.stem}.json"
    output_json.write_text(json.dumps(job.extract_result, indent=2))

    # Toon metadata met citaties en betrouwbaarheidsscores per veld
    if job.extract_metadata and job.extract_metadata.field_metadata:
        document_metadata = job.extract_metadata.field_metadata.document_metadata or {}
        for field, meta in document_metadata.items():
            print(f"{field}: {meta}")
            
