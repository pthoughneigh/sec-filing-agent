"""
config.py

Central configuration for stock_agent. Adjust FILENAMES to add or remove SEC
filings. Pricing constants reflect Claude Haiku 4.5 as of May 2026 — update
them if you switch models.
"""

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