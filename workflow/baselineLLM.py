import datetime
import os
import time

import pandas as pd
from dotenv import load_dotenv

from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# Controleer of de API-sleutel aanwezig is
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(" OPENAI_API_KEY niet gevonden! Controleer je .env bestand.")

# Laad de omgevingsvariabelen in vanuit het .env bestand
load_dotenv(dotenv_path="./../.env", override=True)

# We gebruiken enkel het basis LLM, geen embeddings of vectorstores
llm = LlamaOpenAI(model="gpt-4o", temperature=0, max_tokens=4096)

deep_eval_model = "gpt-4o"

# ==========================================
# 1. EENVOUDIGE BASELINE PROMPT
# ==========================================

# Dit is de enige instructie die het model krijgt, zonder jouw 8 fysiologische regels of tools.
baseline_prompt_template = (
    "Je bent een professionele sportdiëtist. Je geeft voedingsadvies aan sporters.\n"
    "Geef op basis van je eigen kennis een advies met concrete fysiologische richtlijnen "
    "(zoals macro's in grammen en timing) en geef enkele maaltijdsuggesties.\n\n"
    "Vraag van de sporter: {query}\n\n"
    "Antwoord:"
)

# ==========================================
# 2. TESTDATA DEFINIËREN (PERSONA'S)
# ==========================================

agent_questions = [
    # Theorie vraag
    "Ik ga 2 uur sporten, wat zijn de richtlijnen rond koolhydraatinname tijdens de inspanning?",
    # Jogger (lichte/matige inspanning)
    "Ik ben een recreatieve hardloper van 75 kg met een kantoorbaan. Ik ga vanavond van 18:00 tot 19:00 een uur joggen (Lichte intensiteit). Ik ga rond 23:30 slapen en train morgen niet. Wat zijn de exacte fysiologische richtlijnen voor mijn totale inname, en wanneer moet ik wat eten rondom mijn training en bedtijd?",
    # Theorie vraag
    "Wat zijn de richtlijnen voor eiwitinname, inclusief timing en porties?",
    # Fietser (zware inspanning)
    "Ik ben een fietser en weeg 80 kg. Morgen doe ik een zware duurtraining van 2,5 uur (07:00 tot 09:30). Ik ga om 22:00 naar bed. Mijn volgende training is pas de dag erna. Kan je een richtlijn geven voor mijn eiwitten, koolhydraten en de exacte timing vóór, tijdens en voor het slapen?",
    # Triatleet (zeer zware inspanning en snel herstel)
    "Ik ben een triatleet van 65 kg en heb een zeer zware dag. Ik zwem van 06:00 tot 07:30 en loop hard van 12:00 tot 13:00. Mijn bedtijd is 21:30. Hoeveel koolhydraten en eiwitten moet ik in totaal eten, en wat zijn de specifieke herstelregels tussen mijn twee trainingen?",
    # Krachtsporter in de avond (matige inspanning)
    "Mijn doel is spieropbouw. Ik weeg 90 kg en doe van 19:30 tot 21:00 een matig-intensieve krachttraining. Ik slaap om 23:30. Geef me de exacte dagelijkse macro's (eiwitten en koolhydraten) en de timingregels voor mijn maaltijden en voor het slapen.",
    # De Rustdag (lichte inspanning)
    "Ik ben een crossfit-atleet van 60 kg, maar vandaag heb ik een volledige rustdag (Lichte intensiteit). Mijn bedtijd is 22:30. Ik wil mijn maaltijden plannen. Hoeveel eiwit en koolhydraten heb ik vandaag totaal nodig, en hoe moet ik mijn eiwitten over de dag en voor de nacht verdelen?",
    # Zwemmer (korte intensieve inspanning)
    "Ik weeg 70 kg en doe morgenochtend van 06:30 tot 07:15 een intensieve HIIT-zwemtraining (Matige dagelijkse inspanning). Om 22:30 ga ik naar bed. Verder train ik niet. Wat zijn mijn totale macro's en wat zijn de regels voor eten voor, tijdens en na deze korte training?",
    # Ultra-loper (zeer zware inspanning & lange duur)
    "Ik ben een ultra-loper van 68 kg en ga morgen 4 uur lang trailrunnen van 08:00 tot 12:00 (Zeer zware dag). Om 21:30 lig ik in bed. Hoeveel koolhydraten en eiwitten heb ik in totaal nodig, en hoeveel koolhydraten moet ik precies tijdens deze ultra-duursport innemen?",
    # Voetballer (stop-and-go sport)
    "Ik weeg 82 kg en speel vanavond een voetbalwedstrijd van 20:00 tot 21:30 (Zware inspanning). Omdat het laat is, slaap ik pas om 00:00. Wat zijn de precieze richtlijnen voor mijn eiwitten en koolhydraten over de dag, voor de wedstrijd, tijdens de wedstrijd en voor ik ga slapen?",
    # Twee-sessies krachtsporter (snel herstel kracht)
    "Ik ben een krachtsporter van 85 kg op trainingskamp. Ik train van 08:00 tot 09:00 en nog een keer van 14:00 tot 15:00 (Zware dag). Ik slaap om 22:30. Hoeveel moet ik vandaag in totaal eten, en hoe zorg ik voor perfect herstel in die paar uur tussen mijn twee trainingen in?",
    # Yoga-ganger (lichte inspanning)
    "Ik ben een vrouw van 62 kg en start mijn dag met een uur rustige yoga van 07:00 tot 08:00 (Lichte dag). Mijn bedtijd is 22:00. Hoeveel eiwitten en koolhydraten moet ik de hele dag door eten, en is het nodig om tijdens of voor de yoga iets extra's te nemen?",
    # Theorie vraag
    "Hoeveel koolydraten zitten er in een banaan?",
]

ground_truths_agent = [
    # Theorie vraag
    "Consumeer 30 tot 60 gram koolhydraten per uur. Drink dit bij voorkeur als een 6-8% koolhydraat-elektrolyten drankje in kleine slokken elke 10 tot 15 minuten.",
    # Jogger
    "Totaal Eiwit: Je hebt dagelijks tussen de 105 en 150 gram eiwit nodig (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Voor een lichte trainingsdag heb je 225 tot 375 gram koolhydraten nodig (3 tot 5 g/kg). Eiwit per maaltijd: Elke hoofdmaaltijd moet 20 tot 40 gram eiwit bevatten. Eiwitverdeling: Neem deze eiwitrijke maaltijden elke 3 tot 4 uur verspreid over de dag in. Pre-workout: Omdat je training niet langer is dan 60 minuten, is de strikte regel voor pre-workout koolhydraat-loading (1 tot 4 g/kg) hier niet vereist, maar eet gewoon je geplande maaltijd 3 tot 4 uur vooraf. Intra-workout: Tijdens een inspanning van maximaal 60 minuten zijn geen extra koolhydraten nodig. Snel herstel: Niet van toepassing, aangezien je de volgende dag pas weer traint. Pre-sleep: Om 23:00 (30 minuten voor bedtijd) moet je een snack nemen met 30 tot 40 gram eiwit, bij voorkeur caseïne.",
    # Theorie vraag
    "Eiwitten zijn, ongeacht of het een rust- of trainingsdag is, essentieel voor spierherstel en -opbouw, waarbij een totale dagelijkse inname van 1,4 tot 2,0 gram per kilogram lichaamsgewicht voor de meeste sporters volstaat. Voor een maximale stimulatie van de spieropbouw neem je het best porties van 20 tot 40 gram (ongeveer 0,25 gram per kilogram lichaamsgewicht) die je gelijkmatig over de dag verdeelt, bij voorkeur elke 3 tot 4 uur. Besteed daarnaast extra aandacht aan je timing: consumeer hoogwaardige eiwitten vlak voor of direct na je krachttraining voor een robuuste toename in spiergroei, en neem 30 tot 40 gram caseïne-eiwit voor het slapengaan om de nachtelijke spieropbouw te verhogen en je stofwisseling de volgende ochtend te stimuleren.",
    # Fietser
    "Totaal Eiwit: Je hebt dagelijks tussen de 112 en 160 gram eiwit nodig (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Voor een zware trainingsdag heb je 480 tot 800 gram koolhydraten nodig (6 tot 10 g/kg). Eiwit per maaltijd: Elke hoofdmaaltijd moet 20 tot 40 gram eiwit bevatten. Eiwitverdeling: Verspreid deze maaltijden om de 3 tot 4 uur. Pre-workout: Omdat je langer dan 60 minuten traint, moet je 1 tot 4 uur voor de training (tussen 03:00 en 06:00) 80 tot 320 gram koolhydraten (1 tot 4 g/kg) eten. Intra-workout: Omdat de inspanning 150 minuten duurt, moet je 30 tot 60 gram koolhydraten per uur innemen. Snel herstel: Niet van toepassing (voldoende tijd tot de volgende training). Pre-sleep: Om 21:30 neem je een eiwitrijke snack met 30 tot 40 gram eiwit voor spierherstel.",
    # Triatleet
    "Totaal Eiwit: Je hebt dagelijks 91 tot 130 gram eiwit nodig (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Voor een zeer zware trainingsdag heb je 520 tot 780 gram koolhydraten nodig (8 tot 12 g/kg). Eiwit per maaltijd & Verdeling: Elke 3 tot 4 uur moet je een maaltijd met 20 tot 40 gram eiwit eten. Pre-workout: Voor de zwemtraining (langer dan 60 min) moet je 1 tot 4 uur van tevoren 65 tot 260 gram koolhydraten eten (1 tot 4 g/kg). Intra-workout: Tijdens het zwemmen (90 minuten) neem je 30 tot 60 gram koolhydraten per uur. Snel herstel (Cruciaal!): Er zit minder dan 8 uur tussen je zwem- en looptraining (07:30 tot 12:00 is 4,5 uur). Je móét in de eerste 4 uur na het zwemmen elk uur 65 tot 78 gram koolhydraten innemen (1 tot 1,2 g/kg/uur). Pre-sleep: Om 21:00 nuttig je 30 tot 40 gram eiwit voor nachtelijk herstel.",
    # Krachtsporter
    "Totaal Eiwit: Je behoefte ligt tussen de 126 en 180 gram eiwit per dag (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Bij matige inspanning mik je op 450 tot 630 gram koolhydraten per dag (5 tot 7 g/kg). Eiwit per maaltijd & Verdeling: Eet elke 3 tot 4 uur een maaltijd die 20 tot 40 gram eiwit bevat. Pre-workout: Aangezien je 90 minuten traint, dien je tussen 15:30 en 18:30 (1 tot 4 uur vooraf) een inname te doen van 90 tot 360 gram koolhydraten. Intra-workout: Bij deze trainingsduur (tussen 60-150 min) valt een inname van 30 tot 60 gram koolhydraten per uur binnen de richtlijnen. Snel herstel: Niet van toepassing. Pre-sleep: Om 23:00 neem je een snack met 30 tot 40 gram (bij voorkeur langzaam verterend) eiwit.",
    # Rustdag
    "Totaal Eiwit: Ook op een rustdag dien je 84 tot 120 gram eiwit binnen te krijgen voor herstel (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Aangezien het een lichte dag (rustdag) is, heb je 180 tot 300 gram koolhydraten nodig (3 tot 5 g/kg). Eiwit per maaltijd: Consumeer maaltijden die elk 20 tot 40 gram eiwit bevatten. Eiwitverdeling: Eet deze eiwitrijke maaltijden met tussenpozen van 3 tot 4 uur. Pre-, Intra- en Snel Herstel: Deze regels vervallen, aangezien er geen trainingssessie gepland staat. Pre-sleep: Neem om 22:00 uur (30 minuten voor het slapengaan) nog een laatste portie van 30 tot 40 gram eiwit om het spierbehoud tijdens de nacht te optimaliseren.",
    # Zwemmer
    "Totaal Eiwit: Je hebt dagelijks tussen de 98 en 140 gram eiwit nodig (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Voor een matige trainingsdag heb je 350 tot 490 gram koolhydraten nodig (5 tot 7 g/kg). Eiwit per maaltijd: Elke hoofdmaaltijd moet 20 tot 40 gram eiwit bevatten. Eiwitverdeling: Verspreid deze maaltijden om de 3 tot 4 uur. Pre-workout: Omdat je training niet langer is dan 60 minuten, is de strikte pre-workout koolhydraat-loading (1 tot 4 g/kg) niet vereist. Intra-workout: Voor deze korte, intensieve inspanning (45-75 min) kun je een zeer kleine hoeveelheid koolhydraten nemen of simpelweg je mond spoelen met een koolhydraatdrank. Snel herstel: Niet van toepassing. Pre-sleep: Om 22:00 (30 minuten voor bedtijd) neem je een snack met 30 tot 40 gram eiwit, bij voorkeur caseïne.",
    # Ultra-loper
    "Totaal Eiwit: Je hebt dagelijks tussen de 95 en 136 gram eiwit nodig (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Voor een zeer zware trainingsdag heb je 544 tot 816 gram koolhydraten nodig (8 tot 12 g/kg). Eiwit per maaltijd: Elke hoofdmaaltijd moet 20 tot 40 gram eiwit bevatten. Eiwitverdeling: Verspreid deze maaltijden om de 3 tot 4 uur. Pre-workout: Aangezien je 4 uur traint, moet je tussen 04:00 en 07:00 (1 tot 4 uur vooraf) 68 tot 272 gram koolhydraten eten (1 tot 4 g/kg). Intra-workout: Bij ultra-duursport (>2,5-3 uur) mag je tot wel 90 gram koolhydraten per uur innemen. Snel herstel: Niet van toepassing. Pre-sleep: Om 21:00 (30 minuten voor bedtijd) neem je een snack met 30 tot 40 gram eiwit, bij voorkeur caseïne.",
    # Voetballer
    "Totaal Eiwit: Je hebt dagelijks tussen de 115 en 164 gram eiwit nodig (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Voor een zware trainingsdag heb je 492 tot 820 gram koolhydraten nodig (6 tot 10 g/kg). Eiwit per maaltijd: Elke hoofdmaaltijd moet 20 tot 40 gram eiwit bevatten. Eiwitverdeling: Verspreid deze maaltijden om de 3 tot 4 uur. Pre-workout: Omdat je wedstrijd 90 minuten duurt, moet je tussen 16:00 en 19:00 (1 tot 4 uur vooraf) 82 tot 328 gram koolhydraten eten (1 tot 4 g/kg). Intra-workout: Bij een stop-and-go sport van 90 minuten neem je 30 tot 60 gram koolhydraten per uur. Snel herstel: Niet van toepassing. Pre-sleep: Om 23:30 (30 minuten voor bedtijd) neem je een snack met 30 tot 40 gram eiwit, bij voorkeur caseïne.",
    # Twee-sessies krachtsporter
    "Totaal Eiwit: Je hebt dagelijks tussen de 119 en 170 gram eiwit nodig (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Voor een zware trainingsdag heb je 510 tot 850 gram koolhydraten nodig (6 tot 10 g/kg). Eiwit per maaltijd: Elke hoofdmaaltijd moet 20 tot 40 gram eiwit bevatten. Eiwitverdeling: Verspreid deze maaltijden om de 3 tot 4 uur. Pre-workout: Omdat je trainingen exact 60 minuten zijn en niet langer, is de strikte pre-workout koolhydraat-loading niet vereist. Intra-workout: Bij sessies van maximaal 60 minuten zijn geen extra koolhydraten nodig. Snel herstel (Cruciaal!): Omdat er maar 5 uur tussen je trainingen zit (09:00 tot 14:00), moet je in de eerste 4 uur na je eerste training elk uur 85 tot 102 gram koolhydraten innemen (1 tot 1,2 g/kg/uur). Pre-sleep: Om 22:00 (30 minuten voor bedtijd) neem je een snack met 30 tot 40 gram eiwit, bij voorkeur caseïne.",
    # Yoga-ganger
    "Totaal Eiwit: Je hebt dagelijks tussen de 87 en 124 gram eiwit nodig (1,4 tot 2,0 g/kg). Totaal Koolhydraten: Voor een lichte trainingsdag heb je 186 tot 310 gram koolhydraten nodig (3 tot 5 g/kg). Eiwit per maaltijd: Elke hoofdmaaltijd moet 20 tot 40 gram eiwit bevatten. Eiwitverdeling: Verspreid deze maaltijden om de 3 tot 4 uur. Pre-workout: Omdat je training niet langer is dan 60 minuten, is de strikte pre-workout koolhydraat-loading niet vereist. Intra-workout: Tijdens een lichte inspanning van maximaal 60 minuten zijn geen extra koolhydraten nodig. Snel herstel: Niet van toepassing. Pre-sleep: Om 21:30 (30 minuten voor bedtijd) neem je een snack met 30 tot 40 gram eiwit, bij voorkeur caseïne.",
    # Theorie vraag
    "20 gram",
]

# ==========================================
# 3. EVALUATIE LOOP UITVOEREN & DEEPEVAL TESTCASES MAKEN
# ==========================================

agent_logs = []
test_cases = []

for idx, q in enumerate(agent_questions):
    print(f"\n[Test {idx+1}/{len(agent_questions)}] Vraag: {q}")

    # Genereer de prompt voor het basis LLM
    final_prompt = baseline_prompt_template.format(query=q)

    # Vraag het antwoord aan de 'kale' LLM (zonder tools of RAG)
    response = llm.complete(final_prompt)
    actual_output_text = str(response.text)

    thoughts_str = "Geen tools gebruikt (Baseline LLM)."

    # Omdat er geen documenten worden opgehaald, geven we een lege context mee
    current_context = [
        "Geen externe bronnen geraadpleegd. Dit is een Baseline LLM output."
    ]

    # Maak een testcase aan voor DeepEval
    test_case = LLMTestCase(
        input=q,
        actual_output=actual_output_text,
        expected_output=ground_truths_agent[idx],
        retrieval_context=current_context,
    )
    test_cases.append(test_case)

    # Log opslaan (gebruikmakend van Engelse keys voor consistentie)
    print(actual_output_text)
    agent_logs.append(
        {
            "Test_ID": idx + 1,
            "Question_Prompt": q,
            "Thoughts_Tools": thoughts_str,
            "Agent_Final_Answer": actual_output_text,
            "Reference_Answer": ground_truths_agent[idx],
        }
    )

# ==========================================
# 4. DATAFRAME AANMAKEN VOOR LOGS
# ==========================================

df_logs = pd.DataFrame(agent_logs)

# ==========================================
# 5. DEEPEVAL METRICS BEREKENEN & WEERGEVEN
# ==========================================

print("\nDeepEval evaluatie starten")

answer_correctness = GEval(
    name="Answer Correctness",
    criteria=(
        "Vergelijk de fysiologische richtlijnen (aantal grammen eiwitten/koolhydraten en de geadviseerde kloktijden) in de actual_output "
        "met de keiharde feiten in de expected_output. "
        "Als de berekende macro's en tijden wiskundig kloppen met de expected_output, is de score hoog. "
        "Straf het model absoluut NIET af op opmaak (zoals het wel/niet gebruiken van de titels 'DEEL 1' of 'DEEL 2'). "
        "Straf het model ook NIET af op de gekozen recepten of maaltijden, zolang de fysiologische berekening maar juist is."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=deep_eval_model,
    threshold=0.5,
)

metrics = [
    answer_correctness,
    AnswerRelevancyMetric(threshold=0.5, model=deep_eval_model, include_reason=True),
]

print("\n==========================================")
print("EINDRESULTATEN BASELINE EVALUATIE (DEEPEVAL):")
print("==========================================")

for idx, test_case in enumerate(test_cases):
    print(f"\n--- Resultaten Test {idx+1} ---")

    for metric in metrics:
        # geen echte score voor Faithfulness omdat er geen context is
        try:
            metric.measure(test_case)
            score = metric.score
            passed = metric.is_successful()
            reason = metric.reason
        except Exception as e:
            score = 0.0
            passed = False
            reason = (
                f"Fout bij berekenen (waarschijnlijk door ontbreken context): {str(e)}"
            )

        name = metric.__class__.__name__

        print(f"  {name}: {score:.4f} (Geslaagd: {passed})")
        if reason:
            print(f"    Reden: {reason}")

        df_logs.at[idx, f"{name}_Score"] = score
        df_logs.at[idx, f"{name}_Geslaagd"] = passed
        df_logs.at[idx, f"{name}_Reden"] = reason

        time.sleep(5)  # Om OpenAI rate limits te vermijden
    time.sleep(15)

# ==========================================
# 6. LOG OPSLAAN ALS LEESBAAR RAPPORT (.md)
# ==========================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"../output_agents/baseline_evaluatie_rapport_{timestamp}.md"

with open(log_filename, "w", encoding="utf-8") as f:
    f.write("#  Baseline LLM Evaluatie Rapport (Zonder RAG)\n")
    f.write(
        f"**Datum en Tijd:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    f.write("---\n\n")

    for idx, row in df_logs.iterrows():
        f.write(f"## Test {row['Test_ID']}\n\n")
        f.write("###  VRAAG VAN GEBRUIKER\n")
        f.write(f"{row['Question_Prompt']}\n\n")
        
        f.write("###  GEDACHTEN & GEBRUIKTE TOOLS\n")
        f.write(f"{row['Thoughts_Tools']}\n\n")
        
        f.write("###  EINDANTWOORD BASELINE LLM\n")
        f.write(f"{row['Agent_Final_Answer']}\n\n")
        
        f.write("###  REFERENTIE ANTWOORD (GROUND TRUTH)\n")
        f.write(f"{row['Reference_Answer']}\n\n")
        
        f.write("###  DEEPEVAL SCORES\n")
        for m_name in ["GEval", "AnswerRelevancyMetric"]:
            if f"{m_name}_Score" in row:
                f.write(
                    f"**{m_name}:** {row[f'{m_name}_Score']:.4f} (Geslaagd: {row[f'{m_name}_Geslaagd']})\n"
                )
                f.write(f"**Reden:** {row[f'{m_name}_Reden']}\n\n")

        f.write("---\n\n")

    f.write("## GEMIDDELDE SCORES (OVER ALLE TESTEN)\n\n")
    metrics_names = ["GEval", "AnswerRelevancyMetric"]

    for m_name in metrics_names:
        column_name = f"{m_name}_Score"
        if column_name in df_logs.columns:
            average = df_logs[column_name].mean()
            f.write(f"* **{m_name}:** {average:.4f}\n")
        else:
            f.write(f"* **{m_name}:** Niet berekend\n")

    f.write("\n---\n")

# ==========================================
# 7. SLA DE DATA OP VOOR HET ANALYSE SCRIPT
# ==========================================

csv_baseline_path = (
    "../output_agents/baseline_logs.csv"
)
df_logs.to_csv(csv_baseline_path, index=False)
print(
    f"\n Baseline logs succesvol opgeslagen voor latere analyse in: {csv_baseline_path}"
)