import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# ==========================================
# OMGEVING & CONFIGURATIE
# ==========================================

# 1. Laad de API-sleutels in
load_dotenv(dotenv_path="./../.env", override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY niet gevonden! Controleer je .env bestand.")

# 2. Configuratie van modellen (OpenAI)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.embed_model = embed_model
Settings.llm = LlamaOpenAI(model="gpt-5-nano", temperature=0, max_tokens=4096)

# 3. Mappen definiëren (pas de paden aan naar jouw mappenstructuur)
input_directory = "../data/lit_md"
persist_directory = "../data/vectorstore_papers"

# ==========================================
# FASE 1: INLADEN VAN DOCUMENTEN (MAP)
# ==========================================

try:
    # Gebruik input_dir in plaats van input_files om de hele map uit te lezen
    loaded_documents = SimpleDirectoryReader(input_dir=input_directory).load_data()

    if not loaded_documents:
        raise ValueError("Het inladen is mislukt of de map is leeg.")

except Exception as error:
    print(f"Fout bij het laden van de documenten: {error}")
    exit(1)

# ==========================================
# FASE 2: CHUNKING (PIPELINE: MARKDOWN + FALLBACK)
# ==========================================

print("Documenten verwerken in de pijplijn...")

# We bouwen een pijplijn met twee stappen:
# 1. Knippen op logische Markdown koppen
# 2. Een veiligheidsnet: als een Markdown-sectie > 512 tokens is, knip hem dan alsnog op.
pipeline = IngestionPipeline(
    transformations=[
        MarkdownNodeParser(),
        SentenceSplitter(chunk_size=512, chunk_overlap=100),
    ]
)

# Run de documenten door de pijplijn om de veilige nodes te genereren
document_nodes = pipeline.run(documents=loaded_documents)

print(f"✅ {len(document_nodes)} veilige, gestructureerde nodes aangemaakt!")

# ==========================================
# FASE 3: VECTORSTORE BOUWEN & OPSLAAN
# ==========================================

# Bouw de index op basis van ALLE nodes uit ALLE documenten
vector_index = VectorStoreIndex(document_nodes)

# Zorg dat de doelfolder bestaat en sla de opgebouwde index lokaal op
os.makedirs(persist_directory, exist_ok=True)
vector_index.storage_context.persist(persist_dir=persist_directory)

print("✅ Vectorstore succesvol opgeslagen!")