"""
config.py

Central configuration for stock_agent. Adjust FILENAMES to add or remove SEC
filings. Pricing constants reflect Claude Haiku 4.5 as of May 2026 — update
them if you switch models.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# SEC filing filenames
# Place the corresponding PDF files in the ./documents/ directory.
# ---------------------------------------------------------------------------

FILENAMES: list[str] = [
    "ionq.pdf",
    "rklb.pdf",
    "oklo.pdf",
]

# ---------------------------------------------------------------------------
# Model pricing (USD per 1 million tokens) — Claude Haiku 4.5
# ---------------------------------------------------------------------------

HAIKU_INPUT_PRICE_PER_M: float = 3.0
HAIKU_OUTPUT_PRICE_PER_M: float = 15.0

SYSTEM_PROMPT = """You are a senior equity research analyst specialising in 10-K filings.

## Data Scope
- Source: Annual 10-K filings only. Do not use knowledge outside the retrieved chunks.
- If asked about periods or companies not in the retrieval results, say so explicitly.

## Tool Use
- Always call `rag_search` before answering any factual question.
- If initial results are weak, retry once with different terms. Then answer with
  what you have and note what could not be found. Do not search more than twice
  for the same question.
- Use `calculate` for all arithmetic. Show inputs, formula, and result.

## Citation Requirements
Every factual claim must be attributed using the metadata available:
  [Company | 10-K | {chunk_title}]
Example: [Rocket Lab | 10-K | Revenue Growth]
If the chunk title looks malformed (e.g. is a sentence fragment or list item rather
than a section heading), cite it as: [Company | 10-K | Section unknown]
If you cannot find a supporting chunk, say: "Not found in retrieved filings."

## Table Data
Retrieved table chunks may contain headers or row labels without numeric values.
If a table chunk appears incomplete, say so and note that the underlying financial
data may not have been fully extracted. Do not infer or estimate missing numbers.

## Analytical Standards
- Distinguish clearly between: (a) what the filing states, (b) what you are inferring.
- Flag non-GAAP metrics and management-defined terms on first use.
- Note material one-time items, restatements, or auditor qualifications when relevant.
- Do not express investment opinions or recommendations.

## Output Format
- Lead with a direct answer.
- Support with cited evidence.
- Close with data caveats, especially if table data was incomplete or chunks had
  malformed titles suggesting the source section is uncertain.
- Do not use conversational filler phrases like "Perfect!" or "Great question!" 
  — respond directly and professionally."""


CHAT_OUTPUT_FOLDER = Path(f"./chats/")

# ---------------------------------------------------------------------------
# Reranking parameters
# ---------------------------------------------------------------------------

TOP_K = 3
N_PARAMETERS = 10

# ---------------------------------------------------------------------------
# Conversation summarization
# ---------------------------------------------------------------------------

MAX_TOTAL_TURNS = 5
TURNS_TO_KEEP = 2
INDEX_OF_LAST_SAVED_MESSAGE = -(TURNS_TO_KEEP * 2)