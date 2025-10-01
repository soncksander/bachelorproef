# -*- coding: utf-8 -*-
"""
===========================================================
 Qwen3 + alternatieven + INSTRUCTOR → Markdown-rapport
===========================================================

- Leest eerste N recepten uit NDJSON (.jsonl)
- Embed met meerdere modellen (Qwen3 0.6B + BGE/GTE/E5/Mixedbread + INSTRUCTOR)
- Voor INSTRUCTOR: gebruikt (instructie, tekst)-paren zoals README
- Berekent cosine similarity en schrijft een Markdown-rapport (rapport.md)

Installeer:
    pip install -U sentence-transformers transformers scikit-learn accelerate InstructorEmbedding
"""

# === Instellingen ===
PAD_NDJSON = "../data/transformed_recipes.ndjson"  # jouw inputbestand
MODELS = [
    # Qwen3 — 4B & 8B bewust uitgesloten o.b.v. eerdere bevindingen
    # "Qwen/Qwen3-Embedding-4B",   # te groot en te klein verschil t.o.v. 0.6B
    "Qwen/Qwen3-Embedding-0.6B",
    # "Qwen/Qwen3-Embedding-8B",   # te groot en te klein verschil t.o.v. 4B (MTEB)

    # BGE (Engels & modern)
    "BAAI/bge-base-en-v1.5",   # 768-dim
    "BAAI/bge-large-en-v1.5",  # 1024-dim
    "BAAI/bge-m3",             # 1024-dim (multilingual, sterk in EN)

    # GTE (v1.5) - vereist trust_remote_code
    "Alibaba-NLP/gte-base-en-v1.5",   # 768-dim
    "Alibaba-NLP/gte-large-en-v1.5",  # 1024-dim

    # Mixedbread
    "mixedbread-ai/mxbai-embed-large-v1",  # 1024-dim

    # E5
    "intfloat/e5-base-v2",  # 768-dim

    # INSTRUCTOR (HKUNLP / XLANG) — gebruikt instruction + text paren
    "hkunlp/instructor-large",
]
LIMIT = 20                 # aantal recepten om te laden
TOPK = 5                   # top vergelijkbare paren per model in rapport
DEVICE = "auto"            # "auto" | "cpu" | "cuda"
MD_BESTAND = "../data/rapport.md"  # output Markdown

# === Imports ===
import json
import os
import sys
import time
import statistics
from typing import Any, Dict, List, Tuple

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("Installeer: pip install -U sentence-transformers transformers accelerate", file=sys.stderr)
    raise

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    print("Installeer: pip install -U scikit-learn", file=sys.stderr)
    raise

# InstructorEmbedding is apart pakket
try:
    from InstructorEmbedding import INSTRUCTOR
    HAS_INSTRUCTOR = True
except Exception:
    HAS_INSTRUCTOR = False


# === Helpers ===
def detect_device(preferred: str = "auto") -> str:
    """Kies 'cuda' als beschikbaar (bij auto/cuda), anders 'cpu'."""
    if preferred == "cpu":
        return "cpu"
    try:
        import torch  # noqa
        if preferred in ("auto", "cuda") and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def normalize_ingredients(ingredients: Any) -> str:
    """Ingrediënten als lijst -> komma-gescheiden string; anders str()."""
    if ingredients is None:
        return ""
    if isinstance(ingredients, list):
        return ", ".join(str(x) for x in ingredients)
    return str(ingredients)


def recipe_to_text(rec: Dict[str, Any]) -> str:
    """Eenvoudige tekstrepresentatie per recept."""
    title = rec.get("title", "") or ""
    ingredients = normalize_ingredients(rec.get("ingredients", ""))
    instructions = rec.get("instructions", "") or ""
    return f"Titel: {title}\nIngrediënten: {ingredients}\nBereiding: {instructions}"


def load_first_n_recipes(path: str, n: int) -> List[Dict[str, Any]]:
    """Lees de eerste n regels uit NDJSON/JSONL (1 JSON-object per regel)."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Sla ongeldige regels stilletjes over
                pass
    return out


def top_similar_pairs(sim_matrix, k: int) -> List[Tuple[float, int, int]]:
    """Geef top-k paren (i<j) op basis van cosine similarity (excl. diagonaal)."""
    n = sim_matrix.shape[0]
    pairs: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(sim_matrix[i, j]), i, j))
    pairs.sort(key=lambda t: t[0], reverse=True)
    return pairs[:k]


def nearest_neighbors(sim_matrix) -> List[Tuple[int, int, float]]:
    """Voor elk item i: vind j != i met hoogste similarity. Retourneert (i, j, sim)."""
    n = sim_matrix.shape[0]
    out: List[Tuple[int, int, float]] = []
    for i in range(n):
        best_j, best_sim = None, -1.0
        for j in range(n):
            if i == j:
                continue
            s = float(sim_matrix[i, j])
            if s > best_sim:
                best_sim, best_j = s, j
        out.append((i, best_j, best_sim))
    return out


def markdown_escape(text: str) -> str:
    """Kleine escaper voor tabellen: '|' en newlines."""
    if text is None:
        return ""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def needs_trust(model_id: str) -> bool:
    """
    Sommige repos gebruiken custom code.
    - GTE v1.5 vereist trust_remote_code
    - Qwen & mixedbread kunnen dat soms ook gebruiken
    - INSTRUCTOR laadt via apart pakket; hier niet van toepassing
    """
    prefixes = ("Alibaba-NLP/", "Qwen/", "mixedbread-ai/")
    return model_id.startswith(prefixes)


def is_instructor_model(model_id: str) -> bool:
    """Detecteer of we het INSTRUCTOR-pad moeten gebruiken."""
    # INSTRUCTOR modellen leven doorgaans onder hkunlp/ (HKUNLP) en naam bevat 'instructor'
    return "instructor" in model_id.lower()


# === Hoofdprogramma ===
if __name__ == "__main__":
    if not os.path.isfile(PAD_NDJSON):
        print(f"[FOUT] Bestand niet gevonden: {PAD_NDJSON}", file=sys.stderr)
        sys.exit(1)

    recipes = load_first_n_recipes(PAD_NDJSON, LIMIT)
    if not recipes:
        print("[FOUT] Geen geldige recepten gevonden.", file=sys.stderr)
        sys.exit(1)

    texts = [recipe_to_text(r) for r in recipes]
    device = detect_device(DEVICE)

    model_rows = []         # voor de vergelijkingstabel
    per_model_sections = [] # markdown secties per model
    failures = []           # modellen die niet geladen konden worden

    # Standaard INSTRUCTOR-instructie voor recept-documenten (pas aan indien gewenst)
    DOC_INSTRUCTION = "Represent the food recipe for semantic retrieval and clustering:"

    for model_name in MODELS:
        print(f"[INFO] Model laden: {model_name} (device: {device})")
        t0 = time.time()

        try:
            if is_instructor_model(model_name):
                if not HAS_INSTRUCTOR:
                    raise RuntimeError(
                        "InstructorEmbedding niet geïnstalleerd. Doe: pip install InstructorEmbedding"
                    )
                # Laad INSTRUCTOR-model en embed als [instruction, text] paren
                model = INSTRUCTOR(model_name, device=device)
                # Prepare data als [[instruction, text], ...] zoals README
                instructor_inputs = [[DOC_INSTRUCTION, t] for t in texts]
                t_enc0 = time.time()
                embeddings = model.encode(
                    instructor_inputs,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                t_enc1 = time.time()
            else:
                # Standaard SentenceTransformer pad
                model = SentenceTransformer(
                    model_name,
                    device=device,
                    trust_remote_code=needs_trust(model_name)
                )
                t_enc0 = time.time()
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                t_enc1 = time.time()

        except Exception as e:
            print(f"[WAARSCHUWING] Laden/encoden mislukt voor {model_name}: {e}", file=sys.stderr)
            failures.append({"model": model_name, "error": str(e)})
            continue

        enc_secs = t_enc1 - t_enc0

        # Sim matrix
        sim_matrix = cosine_similarity(embeddings)

        # Top paren & NN
        pairs = top_similar_pairs(sim_matrix, TOPK)
        nn = nearest_neighbors(sim_matrix)

        # Samenvattende stats
        nn_sims = [s for (_, _, s) in nn]
        mean_nn = float(statistics.mean(nn_sims)) if nn_sims else 0.0
        median_nn = float(statistics.median(nn_sims)) if nn_sims else 0.0
        mean_topk = float(statistics.mean([p[0] for p in pairs])) if pairs else 0.0
        emb_dim = int(embeddings.shape[1]) if len(embeddings.shape) == 2 else None
        total_secs = time.time() - t0

        model_rows.append({
            "model": model_name,
            "emb_dim": emb_dim,
            "mean_nn": round(mean_nn, 4),
            "median_nn": round(median_nn, 4),
            "mean_topk": round(mean_topk, 4),
            "encode_s": round(enc_secs, 2),
            "total_s": round(total_secs, 2)
        })

        # Markdown sectie (per model)
        top_pairs_lines = [
            "| Rang | Similarity | i | Titel i | j | Titel j |",
            "|-----:|-----------:|--:|---------|--:|---------|",
        ]
        for rank, (sim, i, j) in enumerate(pairs, start=1):
            ti = markdown_escape(recipes[i].get("title", f"<titel {i}>"))
            tj = markdown_escape(recipes[j].get("title", f"<titel {j}>"))
            top_pairs_lines.append(f"| {rank} | {sim:.3f} | {i} | {ti} | {j} | {tj} |")

        nn_lines = []
        for i, j, s in nn:
            ti = markdown_escape(recipes[i].get("title", f"<titel {i}>"))
            tj = markdown_escape(recipes[j].get("title", f"<titel {j}>")) if j is not None else "<none>"
            nn_lines.append(f"- **[{i}] {ti}** → [{j}] {tj}  _(sim={s:.3f})_")

        section = [
            f"## Model: `{model_name}`",
            "",
            f"- Embedding-dimensie: **{emb_dim}**",
            f"- Encodeertijd (alle {len(texts)} recepten): **{enc_secs:.2f}s**",
            f"- Gem. dichtste-buur-similarity: **{mean_nn:.3f}** (mediaan **{median_nn:.3f}**)",
            f"- Gem. top-{TOPK} pair similarity: **{mean_topk:.3f}**",
            "",
            f"### Top {TOPK} vergelijkbare paren",
            *top_pairs_lines,
            "",
            "### Dichtste buur per recept",
            *nn_lines,
            ""
        ]
        per_model_sections.extend(section)

    # Markdown-rapport opbouwen
    header = [
        "# Recept-embeddings sanity check — Qwen3 + alternatieven + INSTRUCTOR",
        "",
        f"- Bestandsbron: `{PAD_NDJSON}`",
        f"- Aantal recepten gebruikt: **{len(texts)}**",
        f"- Top-K paren per model: **{TOPK}**",
        f"- Device: **{device}**",
        "",
        "## Vergelijking tussen modellen",
        "",
        "| Model | Dim | Gem. NN sim | Mediaan NN sim | Gem. Top-K sim | Encode (s) | Totaal (s) |",
        "|-------|----:|------------:|---------------:|---------------:|-----------:|-----------:|",
    ]
    for row in model_rows:
        header.append(
            f"| `{row['model']}` | {row['emb_dim']} | {row['mean_nn']:.4f} | {row['median_nn']:.4f} | "
            f"{row['mean_topk']:.4f} | {row['encode_s']:.2f} | {row['total_s']:.2f} |"
        )

    # Eventuele failures onderaan vermelden
    failure_block = []
    if failures:
        failure_block = [
            "",
            "## Modellen die niet geladen konden worden",
        ]
        for f in failures:
            failure_block.append(f"- `{f['model']}` — fout: `{f['error']}`")

    md_content = "\n".join(header + [""] + per_model_sections + failure_block)

    with open(MD_BESTAND, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[INFO] Markdown-rapport opgeslagen in: {MD_BESTAND}")
    if failures:
        print("[INFO] Sommige modellen konden niet geladen worden. Zie sectie 'Modellen die niet geladen konden worden' in het rapport.")