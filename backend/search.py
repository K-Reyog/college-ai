import os
import re
import json
import math

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "was", "her", "his", "him", "she", "out", "get", "has", "how",
    "now", "see", "two", "way", "who", "did", "its", "let", "put",
    "say", "too", "use", "with", "this", "that", "from", "have",
    "does", "into", "than", "then", "when", "which", "about", "what",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTS_DIR = os.path.join(BASE_DIR, "data", "subjects")


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _index_path(subject):
    return os.path.join(SUBJECTS_DIR, subject, "embeddings_index.json")


def _modules_dir(subject):
    return os.path.join(SUBJECTS_DIR, subject, "extracted_modules")


def _load_index(subject):
    index_path = _index_path(subject)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"No embedding index for subject '{subject}'. Run: python build_index.py {subject}"
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    modules_dir = _modules_dir(subject)
    current_files = set()
    if os.path.isdir(modules_dir):
        current_files = {f for f in os.listdir(modules_dir) if f.endswith(".txt")}

    indexed_files = set(index.get("files", {}).keys())
    if current_files != indexed_files:
        raise FileNotFoundError(
            f"Embedding index for subject '{subject}' is stale (module files changed). "
            f"Run: python build_index.py {subject} --force"
        )

    for name in current_files:
        stat = os.stat(os.path.join(modules_dir, name))
        recorded = index["files"][name]
        if recorded["mtime"] != stat.st_mtime or recorded["size"] != stat.st_size:
            raise FileNotFoundError(
                f"Embedding index for subject '{subject}' is stale (module files changed). "
                f"Run: python build_index.py {subject} --force"
            )

    return index


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a, b):
    norm_a = math.sqrt(_dot(a, a))
    norm_b = math.sqrt(_dot(b, b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _dot(a, b) / (norm_a * norm_b)


def search_modules_semantic(query, subject="cndc", top_n=4):
    index = _load_index(subject)

    response = genai.embed_content(
        model=index.get("model", DEFAULT_EMBEDDING_MODEL),
        content=query,
        task_type="retrieval_query",
        output_dimensionality=index.get("output_dimensionality"),
    )
    query_vector = response["embedding"]

    scored = []
    for chunk in index["chunks"]:
        score = cosine_similarity(query_vector, chunk["embedding"])
        scored.append({
            "file": chunk["file"],
            "chunk_index": chunk["chunk_index"],
            "score": score,
            "text": chunk["text"],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def search_modules_keyword(query, subject="cndc", base_dir=None, top_n=3):
    if base_dir is None:
        base_dir = _modules_dir(subject)

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"No extracted modules found for subject '{subject}' at {base_dir}"
        )

    keywords = {w for w in _tokenize(query) if len(w) > 2 and w not in STOPWORDS}
    if not keywords:
        return []

    results = []

    for file in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, file)
        if not os.path.isfile(path):
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue

        text_lower = text.lower()

        score = 0
        match_positions = []
        for match in re.finditer(r"[a-z0-9]+", text_lower):
            if match.group() in keywords:
                score += 1
                match_positions.append(match.start())

        if score == 0:
            continue

        results.append({
            "file": file,
            "score": score,
            "text": _build_snippet(text, match_positions),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def _build_snippet(text, positions, window_before=500, window_after=1500, max_clusters=2):
    """Extract context around the first match, plus a second far-away match if one exists."""
    chosen = [positions[0]]
    for pos in positions[1:]:
        if pos - chosen[-1] > window_after + window_before:
            chosen.append(pos)
            if len(chosen) >= max_clusters:
                break

    parts = []
    for pos in chosen:
        start = max(0, pos - window_before)
        end = min(len(text), pos + window_after)
        parts.append(text[start:end])

    return "\n...\n".join(parts)


def search_modules(query, subject="cndc", base_dir=None, top_n=4):
    """
    Semantic search by default. A missing/stale index is a setup error and
    propagates uncaught so the caller can tell the user to run build_index.py.
    A live failure embedding the *query* (network/quota) falls back to
    keyword search instead of crashing.
    """
    try:
        return search_modules_semantic(query, subject=subject, top_n=top_n)
    except FileNotFoundError:
        raise
    except Exception as e:
        print(f"[warning] Semantic search unavailable ({e}); falling back to keyword search.")
        return search_modules_keyword(query, subject=subject, base_dir=base_dir, top_n=top_n)
