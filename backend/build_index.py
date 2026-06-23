import os
import json
import argparse
from datetime import datetime

from dotenv import load_dotenv
import google.generativeai as genai

from chunking import normalize_text, chunk_text

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTS_DIR = os.path.join(BASE_DIR, "data", "subjects")

EMBEDDING_MODEL = "models/gemini-embedding-001"
OUTPUT_DIM = 768
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def list_subjects():
    if not os.path.isdir(SUBJECTS_DIR):
        return []
    return sorted(
        name for name in os.listdir(SUBJECTS_DIR)
        if os.path.isdir(os.path.join(SUBJECTS_DIR, name))
    )


def _index_path(subject):
    return os.path.join(SUBJECTS_DIR, subject, "embeddings_index.json")


def _modules_dir(subject):
    return os.path.join(SUBJECTS_DIR, subject, "extracted_modules")


def _is_up_to_date(subject):
    index_path = _index_path(subject)
    modules_dir = _modules_dir(subject)
    if not os.path.isfile(index_path) or not os.path.isdir(modules_dir):
        return False

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    current_files = {f for f in os.listdir(modules_dir) if f.endswith(".txt")}
    indexed_files = set(index.get("files", {}).keys())
    if current_files != indexed_files:
        return False

    for name in current_files:
        stat = os.stat(os.path.join(modules_dir, name))
        recorded = index["files"][name]
        if recorded["mtime"] != stat.st_mtime or recorded["size"] != stat.st_size:
            return False

    return True


def _load_existing_index(subject):
    index_path = _index_path(subject)
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def build_subject_index(subject):
    modules_dir = _modules_dir(subject)
    if not os.path.isdir(modules_dir):
        print(f"No extracted_modules folder for subject '{subject}', skipping.")
        return

    existing = _load_existing_index(subject)
    existing_files = existing.get("files", {}) if existing else {}
    existing_chunks_by_file = {}
    if existing:
        for chunk in existing.get("chunks", []):
            existing_chunks_by_file.setdefault(chunk["file"], []).append(chunk)

    files_meta = {}
    all_chunks = []
    chunk_id = 0
    succeeded, failed, reused = [], [], []

    for filename in sorted(os.listdir(modules_dir)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(modules_dir, filename)
        stat = os.stat(path)

        recorded = existing_files.get(filename)
        if (
            recorded
            and recorded["mtime"] == stat.st_mtime
            and recorded["size"] == stat.st_size
            and filename in existing_chunks_by_file
        ):
            for chunk in existing_chunks_by_file[filename]:
                all_chunks.append({**chunk, "id": chunk_id})
                chunk_id += 1
            files_meta[filename] = recorded
            reused.append(filename)
            print(f"  Reusing cached embeddings for {filename} ({recorded['chunk_count']} chunks)")
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        chunks = chunk_text(normalize_text(raw_text), chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            continue

        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=[c["text"] for c in chunks],
                task_type="retrieval_document",
                output_dimensionality=OUTPUT_DIM,
            )
        except Exception as e:
            print(f"  FAILED {filename}: {e}")
            failed.append(filename)
            continue

        for chunk, vector in zip(chunks, response["embedding"]):
            all_chunks.append({
                "id": chunk_id,
                "file": filename,
                "chunk_index": chunk["index"],
                "text": chunk["text"],
                "embedding": vector,
            })
            chunk_id += 1

        files_meta[filename] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "chunk_count": len(chunks),
        }
        succeeded.append(filename)
        print(f"  Indexed {filename}: {len(chunks)} chunks")

    if not succeeded and not reused:
        print(f"No files indexed for subject '{subject}'. Index not written.")
        return

    index = {
        "subject": subject,
        "model": EMBEDDING_MODEL,
        "output_dimensionality": OUTPUT_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "built_at": datetime.now().isoformat(),
        "files": files_meta,
        "chunks": all_chunks,
    }

    with open(_index_path(subject), "w", encoding="utf-8") as f:
        json.dump(index, f)

    status = "complete" if not failed else "PARTIAL"
    summary = f"Subject '{subject}': {status}. Indexed {len(succeeded)} new, reused {len(reused)} cached file(s)."
    if failed:
        summary += f" FAILED: {', '.join(failed)}. Rerun to retry."
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Build/refresh the semantic search embedding index.")
    parser.add_argument("subject", nargs="?", help="Subject folder name (e.g. cndc)")
    parser.add_argument("--all", action="store_true", help="Rebuild index for every subject")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index looks up to date")
    args = parser.parse_args()

    if args.all:
        subjects = list_subjects()
    elif args.subject:
        subjects = [args.subject]
    else:
        parser.error("Provide a subject name or --all")
        return

    if not subjects:
        print(f"No subjects found under {SUBJECTS_DIR}.")
        return

    for subject in subjects:
        if not args.force and _is_up_to_date(subject):
            print(f"Index already up to date for '{subject}', skipping (use --force to rebuild anyway).")
            continue
        print(f"Building index for '{subject}'...")
        build_subject_index(subject)


if __name__ == "__main__":
    main()
