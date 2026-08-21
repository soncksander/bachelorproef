import datetime
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    PromptTemplate,
    SQLDatabase,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.objects import ObjectIndex, SQLTableSchema
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import SQLTableRetrieverQueryEngine, TransformQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# Controleer of de API-sleutel aanwezig is
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(" OPENAI_API_KEY niet gevonden! Controleer je .env bestand.")

# Laad de omgevingsvariabelen in vanuit het .env bestand
load_dotenv(dotenv_path="./../.env", override=True)

llm = LlamaOpenAI(model="gpt-4o", temperature=0, max_tokens=4096)
deep_eval_model = "gpt-4o"

# ==========================================
# 1. PAPERS INLADEN & ENGINE BOUWEN
# ==========================================

paper_qa_prompt_str = (
    "Je bent een wetenschappelijke sportdiëtist.\n"
    "Hieronder vind je academische contextinformatie uit wetenschappelijke papers.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Lees de specifieke vraag aandachtig, en beantwoord deze kort en feitelijk.\n"
    "STRIKTE REGELS:\n"
    "1. FOCUS: Geef alleen antwoord op de exacte vraag. Als er om grammen wordt gevraagd, geef dan de grammen. Als er om vocht/sportdrank wordt gevraagd, geef dan de drankrichtlijnen.\n"
    "2. FILTER KEIHARD OP TIJDSFASE: Als de vraag over 'tijdens' (during) gaat, negeer dan ABSOLUUT alle data over 'voor' (zoals 30 min before of pre-exercise) én alle data over 'na' (post-exercise / recovery, zoals 1.2 g/kg/h). Geef alléén de data die letterlijk TIJDENS de inspanning geldt.\n"
    "3. HARDE DATA: Haal de specifieke getallen (zoals 30-60 g/h of 6-8%, intervallen in minuten, g/kg lichaamsgewicht) uit de tekst.\n"
    "Vraag: {query_str}\n"
    "Antwoord: "
)

paper_qa_prompt = PromptTemplate(paper_qa_prompt_str)

storage_context_papers = StorageContext.from_defaults(
    persist_dir="../data/vectorstore_papers"
)
paper_index = load_index_from_storage(storage_context_papers)

reranker = LLMRerank(choice_batch_size=5, top_n=3, llm=llm)

base_paper_query_engine = paper_index.as_query_engine(
    similarity_top_k=15,
    node_postprocessors=[reranker],
    text_qa_template=paper_qa_prompt,
)

hyde_transform = HyDEQueryTransform(include_original=True)

paper_query_engine = TransformQueryEngine(
    query_engine=base_paper_query_engine, query_transform=hyde_transform
)

# ==========================================
# 2. RECEPTEN INLADEN & ENGINE BOUWEN
# ==========================================

recipe_qa_prompt_str = (
    "Je bent een uiterst strenge culinaire sportdiëtist.\n"
    "Hieronder vind je contextinformatie uit een receptendatabase (inclusief tags, ingrediënten en instructies).\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Lees de zoekopdracht aandachtig en volg deze stappen.\n"
    "STAP 1: Analyseer de aanvraag. Zoekt de sporter een 'snack' (zoals pre-sleep), een 'breakfast' of een 'dinner/lunch'?\n"
    "STAP 2: CONTROLEER DE TAGS KEIHARD. \n"
    " - Regel A: Als er om een 'snack' of 'pre-sleep' wordt gevraagd, mag je ABSOLUUT GEEN recept selecteren dat de tag 'lunch' of 'dinner' heeft.\n"
    " - Regel B: Als er om een 'breakfast' wordt gevraagd, mag je geen recept met de tag 'dinner' kiezen.\n"
    "STAP 3: VERPLICHTE WEIGERING: Als alle recepten in de context de foute tags hebben voor de vraag, weiger dan te antwoorden. Geef als antwoord EXACT dit: 'Geen passend recept gevonden voor deze specifieke maaltijd.'\n"
    "STAP 4: Als je wél een correct recept vindt dat niet in conflict is met Regel A/B, geef het dan terug in dit exacte format:\n"
    "Title: [Naam]\n"
    "Tags: [Kopieer de exacte tags]\n"
    "Ingredients: [Lijst]\n"
    "Instructions: [Stappen]\n"
    "Vraag: {query_str}\n"
    "Antwoord: "
)

recipe_prompt = PromptTemplate(recipe_qa_prompt_str)

storage_context_recipes = StorageContext.from_defaults(
    persist_dir="../data/vectorstore_recipes_search"
)

recipe_index = load_index_from_storage(storage_context_recipes)

recipe_query_engine = recipe_index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=recipe_prompt,
)

# ==========================================
# 3. SQL VOEDING INLADEN & ENGINE BOUWEN
# ==========================================

engine = create_engine(
    "sqlite:////../data/voeding.db"
)
sql_database = SQLDatabase(engine, include_tables=["Food"])

storage_context_food = StorageContext.from_defaults(
    persist_dir="../data/vectorstore_voeding_rijen"
)
food_rows_index = load_index_from_storage(storage_context_food)
food_rows_retriever = food_rows_index.as_retriever(similarity_top_k=3)
rows_retrievers = {"Food": food_rows_retriever}

table_instruction = (
    "Deze tabel bevat voedingswaarden van producten. "
    "Beschikbare kolommen: 'NEVO-code', 'productnaam', 'energie_kcal', "
    "'koolhydraten_g', 'eiwitten_g', en 'vetten_g'. "
    "Gebruik uitsluitend deze exacte kolomnamen."
)

my_sql_prompt_str = """
Je bent een SQLite expert. Jouw ENIGE taak is om een syntactisch correcte SQLite query te schrijven om de vraag te beantwoorden.
Volg deze regels strikt op:
1. Gebruik UITSLUITEND de tabellen en kolommen uit de schema-informatie.
2. Kopieer de EXACTE productnaam uit de geleverde rij-hints en gebruik deze met de = operator in je WHERE-clausule.
3. Verzin zelf geen productnamen; vertrouw blind op de hints.
4. Geef UITSLUITEND de ruwe SQL-query terug. Gebruik absoluut GEEN markdown-blokken (zoals ```sql), geef GEEN uitleg en GEEN inleidende tekst.

Schema informatie:
{schema}

Vraag: {query_str}
SQLQuery: 
"""

my_sql_prompt = PromptTemplate(my_sql_prompt_str)

table_schema_objs = [SQLTableSchema(table_name="Food", context_str=table_instruction)]
obj_index = ObjectIndex.from_objects(table_schema_objs, index_cls=VectorStoreIndex)

sql_query_engine = SQLTableRetrieverQueryEngine(
    sql_database=sql_database,
    table_retriever=obj_index.as_retriever(similarity_top_k=1),
    rows_retrievers=rows_retrievers,
    text_to_sql_prompt=my_sql_prompt,
    synthesize_response=False,
)

print("Alle Query Engines succesvol geladen! Agent bouwen...")

# ==========================================
# 4. DE OPENAI AGENT & TOOLS BOUWEN
# ==========================================

tool_paper = QueryEngineTool(
    query_engine=paper_query_engine,
    metadata=ToolMetadata(
        name="literatuur_database",
        description=(
            "Gebruik deze tool voor theorievragen over fysiologie en voeding én voor het bepalen van de macro-richtlijnen bij maaltijdplanning. "
            "BELANGRIJK: De papers zijn in het Engels! VERTAAL je zoekterm ALTIJD naar wetenschappelijk Engels. "
            "SPLITS JE ZOEKOPDRACHTEN OP. Als je een dagmenu of wedstrijdvoorbereiding moet berekenen, zoek dan in losse stappen naar de theorie:\n"
            "1. Zoek naar dagelijkse behoefte: bijv. 'daily protein requirements g/kg' en 'daily carbohydrate requirements based on light/moderate/heavy exercise'.\n"
            "2. Zoek naar timing: bijv. 'pre-workout carbohydrate timing and amount g/kg' of 'protein per meal distribution'.\n"
            "3. Zoek naar specifiek herstel/tijdens inspanning: 'carbohydrate intake per hour during exercise' of 'post-exercise fast recovery carbohydrate g/kg/h'.\n"
            "4. Pre-sleep: 'pre-sleep protein requirements timing and amount'.\n"
            "Combineer daarna al deze theorie en reken het zélf uit voor het gewicht van de sporter."
        ),
    ),
)

tool_recipe = QueryEngineTool(
    query_engine=recipe_query_engine,
    metadata=ToolMetadata(
        name="recepten_database",
        description=(
            "Gebruik deze tool om maaltijden te zoeken in de vectorstore. "
            "BELANGRIJK: De recepten zijn opgeslagen met de volgende exacte Engelse tags. Gebruik combinaties van deze tags als je zoekterm:\n"
            "- Meal type (KIES ER EEN): 'breakfast', 'lunch', 'dinner', of 'snack'.\n"
            "- Cuisine: bijv. 'Italian', 'Asian', 'Mexican', 'Global'.\n"
            "- Main ingredients: bijv. 'chicken', 'rice', 'beef', 'oats', 'salmon'.\n"
            "- Additional tags: bijv. 'vegan', 'high-protein', 'quick'.\n"
            "CRUCIALE ZOEKREGEL: Sluit ongewenste maaltijden expliciet uit in je zoekopdracht! "
            "Als je een pre-sleep maaltijd of snack zoekt, voeg dan letterlijk toe: 'snack ONLY, exclude dinner and lunch tags'. "
            "Zoek je een diner, voeg toe: 'dinner ONLY, exclude snack tags'."
        ),
    ),
)

tool_food_sql = QueryEngineTool(
    query_engine=sql_query_engine,
    metadata=ToolMetadata(
        name="Voedingswaarde_database",
        description=(
            "Handig voor het opzoeken van de exacte voedingswaarden (calorieën, eiwitten, koolhydraten, vetten) van ingrediënten. "
            "LET OP 1: Zoek NOOIT naar complete receptnamen. "
            "LET OP 2: Zoek ALTIJD naar concrete, alledaagse basis-voedingsmiddelen in het Nederlands."
        ),
    ),
)

system_prompt = (
    "Je bent een professionele, sterk analytische sportdiëtist en behulpzame AI-assistent. "
    "Lees de vraag van de gebruiker en kies het juiste pad:\n\n"
    "PAD 1: Theorie (bijv. 'wat zijn de richtlijnen rond koolhydraatinname tijdens inspanning?')\n"
    "- Gebruik UITSLUITEND de tool 'literatuur_database'.\n"
    "- STEL NOOIT WEDERVRAGEN aan de gebruiker. Zoek naar de meest complete theorie in de database en geef antwoord.\n"
    "- BLIJF ZOEKEN: Als je eerste zoekopdracht faalt, probeer dan synoniemen (in het Engels) en roep de tool opnieuw aan.\n"
    "- Gebruik géén recepten-tool.\n\n"
    "PAD 2: Praktijk & Maaltijdplanning (bijv. 'Ik weeg 75kg en loop hard om 18:00. Wat moet ik eten?')\n"
    "Als je dit pad kiest, BEN JE VERPLICHT de volgende stappen in deze exacte volgorde uit te voeren:\n"
    "1. HAAL DE THEORIE OP: Gebruik de 'literatuur_database' om de actuele wetenschappelijke richtlijnen (in g/kg) op te zoeken voor:\n"
    "   - Totale dagelijkse inname (eiwit en koolhydraten op basis van trainingsintensiteit).\n"
    "   - Eiwitverdeling (hoeveelheid per maaltijd en timing).\n"
    "   - Pre-workout en intra-workout koolhydraten (afhankelijk van duur van de training).\n"
    "   - Snel herstel (als er <8 uur tussen trainingen zit) en pre-sleep eiwitten.\n"
    "   Roep de tool net zo vaak aan als nodig is om alle macro-regels compleet te krijgen.\n"
    "2. Reken ZELF de regels uit: Vermenigvuldig de gevonden g/kg waarden met het lichaamsgewicht van de sporter. Koppel de theorie aan de opgegeven kloktijden van de sporter.\n"
    "3. ZOEK RECEPTEN: Gebruik de 'recepten_database' (en eventueel Voedingswaarde_database) om 2-3 passende, gevarieerde recepten te zoeken bij de macro's die je zojuist hebt berekend.\n"
    "4. Bouw je eindantwoord EXACT als volgt op:\n"
    "   \n"
    "   DEEL 1: Fysiologische Richtlijnen\n"
    "   [Beschrijf hier de exacte berekende grammen en kloktijden (zoals totale inname, pre/intra/post-workout en pre-sleep) gebaseerd op de theorie die je hebt gevonden.]\n"
    "   \n"
    "   DEEL 2: Maaltijdsuggesties\n"
    "   [Beschrijf hier je gevonden recepten empathisch in vloeiend Nederlands.]\n"
)

agent = OpenAIAgent.from_tools(
    tools=[tool_paper, tool_recipe, tool_food_sql],
    llm=llm,
    system_prompt=system_prompt,
    verbose=True,
    max_function_calls=30,
)

# ==========================================
# 5. TESTDATA DEFINIËREN (PERSONA'S)
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
    # Krachtsporter in de Avond (matige inspanning)
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
# 6. EVALUATIE LOOP UITVOEREN & DEEPEVAL TESTCASES MAKEN
# ==========================================

agent_logs = []
test_cases = []

for index, (question, ground_truth) in enumerate(
    zip(agent_questions, ground_truths_agent)
):

    # Reset het geheugen voor elke nieuwe vraag
    agent.memory.reset()

    print(f"\n Starting Test {index + 1}: {question[:50]}...")

    # De agent verwerkt de routering en het gebruik van tools autonoom
    response = agent.chat(question)

    # Verwerk tools en brokken tekst (chunks)
    thoughts_with_context = []
    thoughts_without_context = []
    current_context = []

    for source in response.sources:
        tool_name = getattr(source, "tool_name", "Unknown Tool")
        tool_input = getattr(source, "raw_input", "No input")
        tool_output = getattr(source, "content", "No output")

        retrieved_chunks = []
        if hasattr(source, "raw_output") and hasattr(source.raw_output, "source_nodes"):
            for node in source.raw_output.source_nodes:
                chunk_text = node.node.get_content().strip()
                retrieved_chunks.append(chunk_text)
                current_context.append(chunk_text)

        base_log_str = (
            f"**TOOL:** {tool_name}\n"
            f"**INPUT:** {tool_input}\n"
            f"**OUTPUT:** {tool_output}\n"
        )

        detailed_log_str = base_log_str
        if retrieved_chunks:
            detailed_log_str += "\n**RETRIEVED CONTEXTS (CHUNKS):**\n"
            for i, chunk in enumerate(retrieved_chunks):
                detailed_log_str += f"> --- Chunk {i + 1} ---\n> {chunk}\n>\n"
        else:
            detailed_log_str += (
                "\n**RETRIEVED CONTEXTS:** No specific chunks found for this tool.\n"
            )

        thoughts_with_context.append(detailed_log_str)
        thoughts_without_context.append(base_log_str)

    # Formatteer de logs als strings
    thoughts_str_with = (
        "\n\n---\n\n".join(thoughts_with_context)
        if thoughts_with_context
        else "No tools used."
    )
    thoughts_str_without = (
        "\n\n---\n\n".join(thoughts_without_context)
        if thoughts_without_context
        else "No tools used."
    )

    if not current_context:
        current_context.append(
            "No external sources consulted (or only the physiologist math tool was used)."
        )

    # Maak de DeepEval TestCase aan
    test_case = LLMTestCase(
        input=question,
        actual_output=str(response.response),
        expected_output=ground_truth,
        retrieval_context=current_context,
    )
    test_cases.append(test_case)

    # Voeg toe aan de logging lijst (gebruik makend van de Engelse keys)
    agent_logs.append(
        {
            "Test_ID": index + 1,
            "Question_Prompt": question,
            "Thoughts_Tools_With": thoughts_str_with,
            "Thoughts_Tools_Without": thoughts_str_without,
            "Agent_Final_Answer": str(response.response),
            "Reference_Answer": ground_truth,
        }
    )

# ==========================================
# 7. DEEPEVAL METRICS BEREKENEN & WEERGEVEN
# ==========================================

# Dataframe aanmaken voor logs
df_logs = pd.DataFrame(agent_logs)

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

custom_faithfulness = GEval(
    name="Custom Faithfulness",
    criteria=(
        "Controleer of de feiten in de actual_output overeenkomen met de retrieval_context. "
        "BELANGRIJK: De retrieval_context is vaak in het Engels, terwijl de actual_output in het Nederlands is. "
        "Straf de actual_output NIET af voor vertalingen. "
        "Straf de actual_output NIET af als recepten (zoals ingrediënten of bereidingswijzen) worden samengevat of in eigen (kortere) bewoordingen worden beschreven, zolang de essentie van het recept maar niet verandert."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model=deep_eval_model,
    threshold=0.5,
)

metrics_dict = {
    "answer_correctness": answer_correctness,
    "custom_faithfulness": custom_faithfulness,
    "FaithfulnessMetric": FaithfulnessMetric(
        threshold=0.5, model=deep_eval_model, include_reason=True
    ),
    "AnswerRelevancyMetric": AnswerRelevancyMetric(
        threshold=0.5, model=deep_eval_model, include_reason=True
    ),
}

print("\n==========================================")
print("EINDRESULTATEN AGENT EVALUATIE (DEEPEVAL):")
print("==========================================")

for idx, test_case in enumerate(test_cases):
    print(f"\n--- Resultaten Test {idx+1} ---")

    # Loop nu door de dictionary heen (key = de naam, value = de metric zelf)
    for m_name, metric in metrics_dict.items():
        metric.measure(test_case)

        score = metric.score
        passed = metric.is_successful()
        reason = metric.reason

        print(f"  {m_name}: {score:.4f} (Geslaagd: {passed})")
        if reason:
            print(f"    Reden: {reason}")

        # Opslaan in dataframe met de correcte naam als prefix
        df_logs.at[idx, f"{m_name}_Score"] = score
        df_logs.at[idx, f"{m_name}_Geslaagd"] = passed
        df_logs.at[idx, f"{m_name}_Reden"] = reason

        time.sleep(5)  # Om OpenAI rate limits te vermijden
    time.sleep(15)

# ==========================================
# 8. LOG OPSLAAN ALS 2 LEESBARE RAPPORTEN (.md)
# ==========================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename_context = f"../output_agents/agent_zonder_fysio__evaluatie_rapport_context_{timestamp}.md"
log_filename_no_context = f"../output_agents/agent_zonder_fysio_evaluatie_rapport_{timestamp}.md"


def write_report(filename, df, tools_column, title):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"#  {title}\n")
        f.write(
            f"**Datum en Tijd:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("---\n\n")

        for idx, row in df.iterrows():
            f.write(f"## Test {row['Test_ID']}\n\n")
            f.write("###  VRAAG VAN GEBRUIKER\n")
            f.write(f"{row['Question_Prompt']}\n\n")

            f.write("###  GEDACHTEN & GEBRUIKTE TOOLS\n")
            f.write(f"{row[tools_column]}\n\n")

            f.write("###  EINDANTWOORD AGENT\n")
            f.write(f"{row['Agent_Final_Answer']}\n\n")

            f.write("###  REFERENTIE ANTWOORD (GROUND TRUTH)\n")
            f.write(f"{row['Reference_Answer']}\n\n")

            f.write("###  DEEPEVAL SCORES\n")
            for m_name in [
                "answer_correctness",
                "custom_faithfulness",
                "FaithfulnessMetric",
                "AnswerRelevancyMetric",
            ]:
                if f"{m_name}_Score" in row:
                    f.write(
                        f"**{m_name}:** {row[f'{m_name}_Score']:.4f} (Geslaagd: {row[f'{m_name}_Geslaagd']})\n"
                    )
                    f.write(f"**Reden:** {row[f'{m_name}_Reden']}\n\n")

            f.write("---\n\n")

        f.write("## GEMIDDELDE SCORES (OVER ALLE TESTEN)\n\n")
        metrics_names = [
            "answer_correctness",
            "custom_faithfulness",
            "FaithfulnessMetric",
            "AnswerRelevancyMetric",
        ]

        for m_name in metrics_names:
            column_name = f"{m_name}_Score"
            if column_name in df.columns:
                average = df[column_name].mean()
                f.write(f"* **{m_name}:** {average:.4f}\n")
            else:
                f.write(f"* **{m_name}:** Niet berekend\n")

        f.write("\n---\n")


# Schrijf het rapport MÉT context
write_report(
    log_filename_context,
    df_logs,
    "Thoughts_Tools_With",
    "Agent Evaluatie Rapport (Met Context)",
)

# Schrijf het rapport ZONDER context
write_report(
    log_filename_no_context,
    df_logs,
    "Thoughts_Tools_Without",
    "Agent Evaluatie Rapport (Zonder Context)",
)

# ==========================================
# SLA DE DATA OP VOOR HET ANALYSE SCRIPT
# ==========================================

csv_rag_path = "../output_agents/rag_zonder_fysio_logs.csv"
df_logs.to_csv(csv_rag_path, index=False)
print(f"\n RAG logs succesvol opgeslagen voor latere analyse in: {csv_rag_path}")