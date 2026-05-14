"""
chunker.py

Converts PDF SEC filings into structured plain-text chunk files that can be
ingested into the vector store.

Each PDF is parsed with `unstructured`, split on Title elements, and written
to ./text_files/<filename>.txt with chunks separated by '\n\n---\n\n'.
Subsequent runs skip already-processed files so re-running is safe.
"""

import logging
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

from config import FILENAMES

log = logging.getLogger("chunker")

# Categories emitted by unstructured that carry meaningful body text
_BODY_CATEGORIES = {"NarrativeText", "Text", "ListItem"}

# Chunk size threshold in characters — when a buffer exceeds this, it is flushed
_CHUNK_SIZE_THRESHOLD = 1_000


def load_pdfs(filenames: list[str]) -> None:
    """Process a list of PDF filenames, chunking each one that has not been
    processed yet. Files that fail are logged and skipped.

    Args:
        filenames: List of PDF filenames (e.g. ['ionq.pdf']) located in
                   ./documents/.
    """
    for filename in filenames:
        load_pdf(filename)


def load_pdf(filename: str) -> bool:
    """Parse a single PDF into structured text chunks and write them to disk.

    Skips processing if a corresponding .txt file already exists in
    ./text_files/.

    The chunking strategy groups body elements (NarrativeText, Text, ListItem)
    under their nearest preceding Title. A new chunk is also started whenever
    the accumulated text exceeds _CHUNK_SIZE_THRESHOLD characters.

    Args:
        filename: PDF filename (basename only), e.g. 'ionq.pdf'. The file must
                  exist at ./documents/<filename>.

    Returns:
        True if chunking succeeded or was skipped (already done), False on error.
    """
    output_path = Path(f"./text_files/{filename}.txt")
    if output_path.exists():
        log.info("%s already chunked — skipping.", filename)
        return True

    pdf_path = Path(f"./documents/{filename}")
    if not pdf_path.exists():
        log.error(
            "'%s' not found. Place the PDF in the ./documents/ folder and try again.",
            pdf_path,
        )
        return False

    log.info("Parsing %s …", filename)
    try:
        elements = partition_pdf(
            str(pdf_path),
            strategy="fast",
            languages=["eng"],
        )
    except Exception as exc:
        log.error("Failed to parse '%s': %s", filename, exc)
        return False

    chunks: list[str] = []
    current_title = ""
    buffer: list[str] = []

    for element in elements:
        category: str = element.category

        # Skip structural noise
        if category in {"Header", "Footer", "UncategorizedText"}:
            continue
        if category == "Title" and element.text.lower() == "table of contents":
            continue

        if category == "Title":
            # Flush the current buffer before starting a new section
            if buffer:
                chunks.append(f"{current_title}\n" + "\n".join(buffer))
                buffer = []
            # FIX 1: When consecutive titles appear with no body between them,
            # concatenate rather than silently overwriting the previous title.
            # Without this, the earlier title is lost entirely because the
            # `if buffer` guard above skips the flush when buffer is empty.
            if current_title and not buffer:
                current_title = f"{current_title} — {element.text}"
            else:
                current_title = element.text

        elif category in _BODY_CATEGORIES:
            buffer.append(element.text)
            # Flush when the buffer grows large enough
            if len("\n".join(buffer)) > _CHUNK_SIZE_THRESHOLD:
                chunks.append(f"{current_title}\n" + "\n".join(buffer))
                buffer = []
                # Reset title after a mid-section flush so the next chunk
                # doesn't re-use a stale heading from a previous section.
                current_title = ""

    # Flush any remaining content
    if buffer:
        chunks.append(f"{current_title}\n" + "\n".join(buffer))

    if not chunks:
        log.warning(
            "No text extracted from '%s'. "
            "The PDF may be scanned/image-based and require OCR.",
            filename,
        )
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")
    except OSError as exc:
        log.error("Could not write '%s': %s", output_path, exc)
        return False

    log.info("%s → %d chunks written to %s", filename, len(chunks), output_path)
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    load_pdfs(FILENAMES)