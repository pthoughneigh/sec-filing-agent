"""
ingest.py

Embeds pre-chunked text files and upserts them into a ChromaDB persistent
vector store. Uses all-mpnet-base-v2 model for embeddings.

Shared module-level objects (`collection`, `model`) are imported by agent.py so
that the same model and collection are reused at query time.
"""

import logging
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import FILENAMES
log = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Shared clients — imported by agent.py
# ---------------------------------------------------------------------------

try:
    model = SentenceTransformer('all-mpnet-base-v2')
except Exception as exc:
    log.error("Could not load embedding model: %s", exc)
    sys.exit(1) # Fatal — agent cannot function without the embedding model.

try:
    _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = _chroma_client.get_or_create_collection(name="stock_filings")
except Exception as exc:
    log.error("Could not open ChromaDB at ./chroma_db: %s", exc)
    log.error("If the database is corrupted, delete the chroma_db/ folder and re-run.")
    sys.exit(1) # Fatal — agent cannot function without the database.


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


def _split_title(chunk: str) -> tuple[str, str]:
    """Return (title, body) from a chunk, handling missing titles gracefully."""
    lines = chunk.split("\n", 1)
    if len(lines) == 2:
        return lines[0].strip(), lines[1].strip()
    return "", chunk.strip()


def ingest(filename: str) -> bool:
    """Embed and store all chunks from a single text file into ChromaDB.

    Skips ingestion if documents from this source are already present in the
    collection, making the operation idempotent.

    Args:
        filename: PDF basename (e.g. 'ionq.pdf'). The corresponding chunked
                  text file must already exist at ./text_files/<filename>.txt
                  (produced by chunker.load_pdf).

    Returns:
        True if ingestion succeeded or was skipped, False on error.
    """
    source_path = f"./documents/{filename}"

    try:
        existing = collection.get(where={"source": source_path})
        if existing["ids"]:
            log.info("%s already ingested — skipping.", filename)
            return True
    except Exception as exc:
        log.error("Could not query collection for '%s': %s", filename, exc)
        return False

    text_path = f"./text_files/{filename}.txt"
    try:
        with open(text_path, encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        log.error(
            "'%s' not found. Chunking may have failed for '%s' — check the output above.",
            text_path,
            filename,
        )
        return False
    except OSError as exc:
        log.error("Could not read '%s': %s", text_path, exc)
        return False

    # Filter out whitespace-only chunks at split time so they are never
    # embedded or stored.
    chunks: list[str] = [c for c in raw.split("\n\n---\n\n") if c.strip()]
    if not chunks:
        log.warning("'%s' is empty — skipping.", text_path)
        return False

    # Split title from body so it can be stored as metadata for citations.
    titles_and_bodies = [_split_title(c) for c in chunks]

    log.info("Embedding %s (%d chunks) …", filename, len(chunks))
    try:
        embeddings: list[list[float]] = model.encode(chunks).tolist()
    except Exception as exc:
        log.error("Embedding failed for '%s': %s", filename, exc)
        return False

    # Use Path.stem instead of split(".")[0] so that filenames containing
    # multiple dots produce the correct stem rather than just the first segment,
    # which could cause ID collisions between different filings.
    stem = Path(filename).stem
    ids = [f"{stem}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": source_path,
            "chunk_index": i,
            "title": title,
        }
        for i, (title, _) in enumerate(titles_and_bodies)
    ]

    try:
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
    except Exception as exc:
        log.error("Could not write to ChromaDB for '%s': %s", filename, exc)
        return False

    log.info("%s → %d chunks added to collection.", filename, len(chunks))
    return True


def ingest_all(filenames: list[str]) -> None:
    """Ingest every filename in the provided list. Failed files are logged
    and skipped so the rest of the list still processes.

    Args:
        filenames: List of PDF basenames to process.
    """
    for filename in filenames:
        ingest(filename)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ingest_all(FILENAMES)