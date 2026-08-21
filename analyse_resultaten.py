import os

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# ==========================================
# 0. SETUP & API KEY
# ==========================================

# Laad de omgevingsvariabelen in vanuit het .env bestand
load_dotenv(dotenv_path="./../.env", override=True)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY niet gevonden!")

# Initialiseer het LLM
llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)

# ==========================================
# 1. PYDANTIC EXTRACTOR DEFINIËREN (NU MET INTRA-WORKOUT)
# ==========================================

class ExtractedMacros(BaseModel):
    protein_min: float
    protein_max: float
    carbs_min: float
    carbs_max: float
    intra_carb_min: float
    intra_carb_max: float


extraction_prompt = (
    "Je bent een zeer precieze data-extractor voor sportvoeding.\n"
    "Haal de minimum en maximum waarden uit de onderstaande tekst voor de volgende 3 categorieën:\n"
    "1. Totaal Eiwit per dag (in grammen)\n"
    "2. Totaal Koolhydraten per dag (in grammen)\n"
    "3. Intra-workout Koolhydraten (tijdens de inspanning, in grammen per uur).\n\n"
    "REGELS:\n"
    "- Als er een range staat (bijv. 100 tot 150), vul dan 100 in bij min en 150 bij max.\n"
    "- Als er maar ÉÉN exact getal staat, vul dat in bij ZOWEL min als max.\n"
    "- Als er bij intra-workout staat 'niet nodig', '0 gram', of het wordt simpelweg NIET genoemd, vul dan 0 in bij min én max.\n"
    "- Haal de getallen eruit, ongeacht of het een tabel of doorlopende tekst is.\n\n"
    "Tekst: {text}\n"
)

# Maak de extractor aan via de LlamaIndex TextCompletionProgram
extractor = LLMTextCompletionProgram.from_defaults(
    output_cls=ExtractedMacros, prompt_template_str=extraction_prompt, llm=llm
)

# ==========================================
# 2. DATA INLADEN
# ==========================================

print("📥 CSV bestanden inladen...")
path_baseline = "../output_agents/baseline_logs.csv"
path_rag = "../output_agents/rag_logs.csv"

# Controleer of bestanden bestaan
if not os.path.exists(path_baseline) or not os.path.exists(path_rag):
    raise FileNotFoundError(
        "❌ CSV bestanden niet gevonden! Run eerst je baseline- en RAG-script."
    )

# Laad de dataframes in
df_base = pd.read_csv(path_baseline)
df_rag = pd.read_csv(path_rag)

all_evaluation_data = []

# ==========================================
# 3. DATA EXTRAHEREN UIT DE TEKSTEN
# ==========================================

print("🔍 Starten met extraheren van macro's (Dagelijks + Intra-workout)...")

for idx in range(len(df_base)):
    
    # Haal de Ground Truth tekst op
    gt_text = str(df_base.loc[idx, "Reference_Answer"])

    # We pakken alleen de testcases die fysiologische berekeningen bevatten
    if "Totaal Eiwit" in gt_text:
        print(f"  -> Verwerken van Test {idx+1}...")

        base_text = str(df_base.loc[idx, "Agent_Final_Answer"])
        
        # Bepaal de juiste kolomnaam voor RAG, afhankelijk van hoe het script is opgeslagen
        rag_col_name = (
            "Agent_Final_Answer"
            if "Agent_Final_Answer" in df_rag.columns
            else "Eindantwoord_Agent"
        )
        rag_text = str(df_rag.loc[idx, rag_col_name])

        try:
            # Voer de extractie uit voor de drie soorten teksten
            gt_macros = extractor(text=gt_text)
            base_macros = extractor(text=base_text)
            rag_macros = extractor(text=rag_text)

            # --- Dagelijkse Eiwitten ---
            all_evaluation_data.append(
                {
                    "Testcase": f"Test {idx+1}",
                    "Macronutrient": "Totaal Eiwit (Dag)",
                    "Target_Min": gt_macros.protein_min,
                    "Target_Max": gt_macros.protein_max,
                    "Baseline_Output": (base_macros.protein_min + base_macros.protein_max) / 2,
                    "RAG_Output": (rag_macros.protein_min + rag_macros.protein_max) / 2,
                }
            )

            # --- Dagelijkse Koolhydraten ---
            all_evaluation_data.append(
                {
                    "Testcase": f"Test {idx+1}",
                    "Macronutrient": "Totaal Carbs (Dag)",
                    "Target_Min": gt_macros.carbs_min,
                    "Target_Max": gt_macros.carbs_max,
                    "Baseline_Output": (base_macros.carbs_min + base_macros.carbs_max) / 2,
                    "RAG_Output": (rag_macros.carbs_min + rag_macros.carbs_max) / 2,
                }
            )

            # --- Intra-workout Koolhydraten (Tijdens sporten) ---
            all_evaluation_data.append(
                {
                    "Testcase": f"Test {idx+1}",
                    "Macronutrient": "Intra-workout Carbs/Uur",
                    "Target_Min": gt_macros.intra_carb_min,
                    "Target_Max": gt_macros.intra_carb_max,
                    "Baseline_Output": (base_macros.intra_carb_min + base_macros.intra_carb_max) / 2,
                    "RAG_Output": (rag_macros.intra_carb_min + rag_macros.intra_carb_max) / 2,
                }
            )

        except Exception as e:
            print(f"  ❌ Fout bij extractie Test {idx+1}: {e}")

# ==========================================
# 4. AFWIJKINGEN BEREKENEN EN GRAFIEKEN MAKEN
# ==========================================

if len(all_evaluation_data) > 0:
    print("\n📊 Genereren van tabellen en grafieken...")
    df_plot = pd.DataFrame(all_evaluation_data)

    def calculate_deviation(value, minimum, maximum):
        """
        Berekent de afwijking. Als de waarde binnen min en max valt is de afwijking 0.
        """
        if pd.isna(value):
            return None
        if value < minimum:
            return value - minimum
        elif value > maximum:
            return value - maximum
        else:
            return 0

    # Pas de afwijkingsformule toe op de dataframe
    df_plot["Deviation_Baseline"] = df_plot.apply(
        lambda row: calculate_deviation(
            row["Baseline_Output"], row["Target_Min"], row["Target_Max"]
        ),
        axis=1,
    )
    df_plot["Deviation_RAG"] = df_plot.apply(
        lambda row: calculate_deviation(
            row["RAG_Output"], row["Target_Min"], row["Target_Max"]
        ),
        axis=1,
    )

    # Tabel printen in de console
    print("\n=== QUANTITATIVE NUTRITIONAL DEVIATION TABLE ===")
    print(
        df_plot[
            [
                "Testcase",
                "Macronutrient",
                "Target_Min",
                "Target_Max",
                "Deviation_Baseline",
                "Deviation_RAG",
            ]
        ].to_string(index=False)
    )

    # Tabel opslaan als CSV
    csv_path = "../output_agents/deviatie_tabel.csv"
    df_plot.to_csv(csv_path, index=False)

    # ---------------------------------------------------------
    # 📈 GRAFIEK 1: DAGELIJKSE MACRO'S (Eiwit & Koolhydraten)
    # ---------------------------------------------------------
    mask_daily = df_plot["Macronutrient"].isin(["Totaal Eiwit (Dag)", "Totaal Carbs (Dag)"])
    df_daily = df_plot[mask_daily].copy()

    # Kortere labels maken (met de 'r' voor de raw string warning)
    df_daily["Label"] = (
        df_daily["Testcase"]
        + " ("
        + df_daily["Macronutrient"]
        .str.replace("Totaal ", "")
        .str.replace(r" \(Dag\)", "", regex=True)
        + ")"
    )

    df_chart_1 = df_daily.set_index("Label")[["Deviation_Baseline", "Deviation_RAG"]]

    # Hernoem de kolommen zodat Pandas zelf de legenda perfect maakt
    df_chart_1 = df_chart_1.rename(
        columns={
            "Deviation_Baseline": "Baseline LLM (Foutmarge)",
            "Deviation_RAG": "RAG Agent (Foutmarge)",
        }
    )

    plt.figure()
    ax1 = df_chart_1.plot(
        kind="bar", figsize=(14, 6), color=["#e74c3c", "#2ecc71"], edgecolor="black"
    )
    plt.title(
        "Grafiek 1: Afwijking in DAGELIJKSE Macro's (Baseline vs RAG)",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Afwijking (gram)", fontsize=12)
    plt.xlabel("Testcase & Macronutriënt", fontsize=12)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()  # Zorgt dat er niks buiten beeld valt, plt.legend() is overbodig

    path_chart_1 = "../output_agents/grafiek_1_dagelijks.png"
    plt.savefig(path_chart_1, dpi=300)
    plt.close("all")

    # ---------------------------------------------------------
    # 📈 GRAFIEK 2: INTRA-WORKOUT KOOLHYDRATEN
    # ---------------------------------------------------------
    mask_intra = df_plot["Macronutrient"] == "Intra-workout Carbs/Uur"
    df_intra = df_plot[mask_intra].copy()

    df_chart_2 = df_intra.set_index("Testcase")[["Deviation_Baseline", "Deviation_RAG"]]

    # Hernoem de kolommen zodat Pandas zelf de legenda perfect maakt
    df_chart_2 = df_chart_2.rename(
        columns={
            "Deviation_Baseline": "Baseline LLM (Foutmarge)",
            "Deviation_RAG": "RAG Agent (Foutmarge)",
        }
    )

    plt.figure()
    ax2 = df_chart_2.plot(
        kind="bar", figsize=(10, 6), color=["#e74c3c", "#2ecc71"], edgecolor="black"
    )
    plt.title(
        "Grafiek 2: Afwijking in INTRA-WORKOUT Koolhydraten (tijdens sporten)",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Afwijking (gram per uur)", fontsize=12)
    plt.xlabel("Testcase", fontsize=12)
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path_chart_2 = "../output_agents/grafiek_2_intraworkout.png"
    plt.savefig(path_chart_2, dpi=300)
    plt.close("all")

    print(f"\n✅ Resultaten succesvol opgeslagen!")
    print(f"   -> {path_chart_1}")
    print(f"   -> {path_chart_2}")

else:
    print("\nGeen data verwerkt.")