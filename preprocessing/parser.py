# Installeer het package met: pip install llama-cloud (bij voorkeur in een aparte virtuele omgeving)

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import LlamaCloud

# Laad de omgevingsvariabelen in vanuit het .env bestand
load_dotenv(
    dotenv_path="../.env", override=True
)

# Controleer of de API-sleutel aanwezig is
if not os.getenv("LLAMA_CLOUD_API_KEY"):
    raise ValueError("LLAMA_CLOUD_API_KEY niet gevonden.")


def parser(INPUT_DIR, OUTPUT_MD_DIR, instruction):
    # Maak de outputmap aan als deze nog niet bestaat
    Path(OUTPUT_MD_DIR).mkdir(parents=True, exist_ok=True)

    # Initialiseer de LlamaCloud client
    client = LlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

    # Overloop alle PDF-bestanden in de inputmap
    for pdf_path in Path(INPUT_DIR).glob("*.pdf"):
        print(f"Bezig met verwerken van: {pdf_path.name} ...")

        # Bepaal de output bestandsnaam op basis van de originele bestandsnaam (zonder .pdf)
        # pdf_path.stem geeft de bestandsnaam zonder de extensie (bijv. "protein_and_exercise")
        output_md = Path(OUTPUT_MD_DIR) / f"{pdf_path.stem}.md"

        if output_md.exists():
            print(f"Overslaan: {pdf_path.name} is al verwerkt (bestand bestaat al).")
            continue

        try:
            with open(pdf_path, "rb") as f:
                # Bestand uploaden
                file_obj = client.files.create(file=pdf_path, purpose="parse")

            # Indienen, status controleren (polling) en ophalen (parsing.parse combineert create / wait_for_completion / get)
            # Werpt een fout op bij FAILED of CANCELLED. Pas indien nodig polling_interval= en timeout= aan.
            result = client.parsing.parse(
                file_id=file_obj.id,
                # Het parse-niveau. Opties: fast, cost_effective, agentic, agentic_plus
                tier="agentic",
                # De te gebruiken versie van het parse-niveau. Gebruik 'latest' voor de meest recente versie.
                version="latest",
                agentic_options={"custom_prompt": instruction},
                # expand: welke velden geëxtraheerd moeten worden (markdown_full, text_full, items, *_content_metadata, ...)
                expand=["markdown_full"],
            )

            # Sla de markdown lokaal op
            Path(output_md).write_text(result.markdown_full or "")
            print(f"{len(result.markdown_full or '')} tekens aan markdown weggeschreven")

        except Exception as e:
            print(
                f"Er is een fout opgetreden bij het verwerken van {pdf_path.name}: {e}\n"
            )

    print("Alle bestanden zijn verwerkt!")


# Paden naar de in te lezen PDF's
path_recipes_pdf = "../data/recipes_van_bij_ons"
path_papers_pdf = "../data/literature"

# Stap 1: Alle recepten en alle papers parsen naar markdown en opslaan
output_markdown_dir_recipes = (
    "../data/rec_md"
)
output_markdown_dir_papers = (
    "../data/lit_md"
)

# Instructies voor de agentic parser
recipe_instruction = (
    "Dit is een recept. Formatteer het als een gestructureerd document. "
    "Maak duidelijk onderscheid tussen de 'Titel', 'Voorbereidingstijd', "
    "'Ingrediënten' en de 'Instructies/Bereidingswijze'. "
    "Zorg dat alle stappen logisch genummerd zijn."
)

paper_instruction = (
    "Dit is een wetenschappelijke paper. Formatteer het als een overzichtelijk, "
    "gestructureerd document met behoud van de originele sectie-hiërarchie. "
    "Maak duidelijk onderscheid tussen standaardsecties zoals 'Titel', 'Abstract', "
    "'Introductie', 'Methodologie', 'Resultaten', 'Discussie', 'Conclusie' en 'Referenties'. "
    "Zorg ervoor dat alle tabellen exact en overzichtelijk worden geformatteerd als Markdown-tabellen. "
    "Wanneer er grafieken, diagrammen of figuren in de tekst staan, voeg dan een "
    "duidelijke, uitgebreide tekstuele beschrijving toe van wat de figuur afbeeldt en de belangrijkste inzichten ervan. "
    "Behoud wiskundige formules indien aanwezig."
)

# Functie aanroepen om recepten te parsen
parser(
    path_recipes_pdf,
    output_markdown_dir_recipes,
    recipe_instruction,
)