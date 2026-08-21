import json
import os

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# ==========================================
# OMGEVING & CONFIGURATIE
# ==========================================

# Laad de omgevingsvariabelen in vanuit het .env bestand
load_dotenv(dotenv_path="../.env", override=True)

# Controleer of de API-sleutel aanwezig is
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(" OPENAI_API_KEY niet gevonden! Controleer je .env bestand.")

# Configuratie van modellen (OpenAI)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4o", temperature=0, max_tokens=4096)
Settings.embed_model = embed_model

# Zorg dat dit pad verwijst naar de map met je nieuwe JSON bestanden
input_directory = "../data/rec_extracted"
persist_dir = "../data/vectorstore_recipes_search"

# ==========================================
# FASE 1: JSON INLADEN & NODES BOUWEN
# ==========================================

guided_nodes = []
processed_files = 0
no_tags = 0
names_no_tags = []

for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        filepath = os.path.join(input_directory, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f" Fout bij inladen van {filename}, bestand overgeslagen.")
                continue

        # 1. Haal titel en tags op
        title = data.get("title", "Onbekend Recept")
        tags_dict = data.get("tags", {})  # Dit is nu een dictionary!

        if not tags_dict:
            no_tags += 1
            names_no_tags.append(title)
            continue

        # 1b. Sla de dictionary plat naar een lijst van strings
        flat_tags = []

        # .get() retourneert None als de sleutel niet bestaat.
        # Door "if tags_dict.get(...):" te gebruiken, filter je None, lege strings en lege lijsten er direct uit!
        if tags_dict.get("meal_type"):
            flat_tags.append(tags_dict["meal_type"])

        if tags_dict.get("cuisine"):
            flat_tags.append(tags_dict["cuisine"])

        if tags_dict.get("main_ingredients"):
            flat_tags.extend(tags_dict["main_ingredients"])

        if tags_dict.get("additional_tags"):
            flat_tags.extend(tags_dict["additional_tags"])

        tags_str = ", ".join(flat_tags)

        # 2. De 'Target Vector' (Alleen titel en de platgeslagen tags)
        embed_text = f"Titel: {title}\nTags: {tags_str}"

        # 3. Bouw de string voor de LLM
        recipe_text = f"Recept: {title}\n"
        recipe_text += f"Beschrijving: {data.get('description', '')}\n"
        recipe_text += f"Bereidingstijd: {data.get('prep_time', 'Onbekend')}\n\n"  # 'min' weggelaten

        recipe_text += "Ingrediënten:\n"
        for ing in data.get("ingredients", []):
            quantity_str = f"{ing.get('quantity') or ''} {ing.get('unit') or ''}".strip()
            name_str = ing.get("name", "")
            preparation_str = f" ({ing.get('preparation')})" if ing.get("preparation") else ""
            recipe_text += f"- {quantity_str} {name_str}{preparation_str}\n"

        recipe_text += "\nInstructies:\n"
        for inst in data.get("instructions", []):
            recipe_text += f"{inst.get('step_number')}. {inst.get('text')}\n"

        # 4. Maak de TextNode aan
        node = TextNode(text=embed_text)

        # 5. Metadata toewijzen (we slaan de platte lijst op als 'Tags')
        node.metadata = {"Full_recipe": recipe_text, "Tags": flat_tags}

        # 6. CRUCIAAL: Excluded keys correct gespeld!
        # Zo weet LlamaIndex 100% zeker dat hij "Full_recipe" NIET meeneemt in de wiskundige berekening.
        node.excluded_embed_metadata_keys = ["Full_recipe", "Tags"]

        guided_nodes.append(node)
        processed_files += 1

print(f"Er zijn {processed_files} JSON-recepten succesvol omgezet naar Nodes.")

# ==========================================
# FASE 2: VECTORSTORE BOUWEN & OPSLAAN
# ==========================================

print("Vectorstore wordt opgebouwd op basis van titels en tags...")

# We bouwen de index direct vanuit onze handgemaakte nodes
index = VectorStoreIndex(guided_nodes)

# Sla de index lokaal op
os.makedirs(persist_dir, exist_ok=True)
index.storage_context.persist(persist_dir=persist_dir)

print(f"Bestanden zonder tags: {no_tags}")
print(f"Namen van bestanden zonder tags: {names_no_tags}")
print(f" Vectorstore succesvol opgebouwd en opgeslagen in {persist_dir}")