"""
ingest.py

Embeds pre-chunked text files and upserts them into a ChromaDB persistent
vector store. Uses VoyageAI's voyage-3 model for embeddings.

Shared module-level objects (`collection`, `vo`) are imported by agent.py so
that the same client and collection are reused at query time.
"""

import logging
import sys
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

from config import FILENAMES

log = logging.getLogger("ingest")

load_dotenv()

# ---------------------------------------------------------------------------
# Shared clients — imported by agent.py
# ---------------------------------------------------------------------------

try:
    vo = voyageai.Client()
except Exception as exc:
    log.error("Could not initialise VoyageAI client: %s", exc)
    log.error("Make sure VOYAGE_API_KEY is set in your .env file.")
    sys.exit(1)

try:
    _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = _chroma_client.get_or_create_collection(name="stock_filings")
except Exception as exc:
    log.error("Could not open ChromaDB at ./chroma_db: %s", exc)
    log.error("If the database is corrupted, delete the chroma_db/ folder and re-run.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


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

    # FIX 2: Filter out whitespace-only chunks at split time so they are never
    # embedded or stored. The old `not any(chunks)` check passed as long as a
    # single non-empty chunk existed, meaning blank chunks could still slip
    # through into ChromaDB.
    chunks: list[str] = [c for c in raw.split("\n\n---\n\n") if c.strip()]
    if not chunks:
        log.warning("'%s' is empty — skipping.", text_path)
        return False

    log.info("Embedding %s (%d chunks) …", filename, len(chunks))
    try:
        embeddings: list[list[float]] = vo.embed(chunks, model="voyage-3").embeddings
    except Exception as exc:
        log.error("VoyageAI embedding failed for '%s': %s", filename, exc)
        return False

    # FIX 1: Use Path.stem instead of split(".")[0] so that filenames
    # containing multiple dots (e.g. 'ionq.10k.pdf') produce the correct stem
    # ('ionq.10k') rather than just the first segment ('ionq'), which could
    # cause ID collisions between different filings.
    stem = Path(filename).stem
    ids = [f"{stem}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_path, "chunk_index": i} for i in range(len(chunks))]

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