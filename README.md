# sec_filing_agent


A local CLI agent that lets you ask natural-language questions about SEC filings using a RAG (Retrieval-Augmented Generation) pipeline powered by the Anthropic API, VoyageAI embeddings, and ChromaDB.

---

## Features

- **RAG search** — semantic search over chunked SEC filings (10-K, 10-Q, etc.)
- **Calculator tool** — the model can perform arithmetic on figures it retrieves
- **Multi-turn conversation** — full context is preserved across turns in one session
- **Idempotent ingestion** — re-running the agent never re-embeds already-processed files
- **Token cost tracking** — session summary printed on exit

---

## Project Structure

```
stock_agent/
├── src/
│   ├── agent.py        # Agentic loop, tool dispatch, CLI entry point
│   ├── chunker.py      # PDF → structured text chunks
│   ├── ingest.py       # Embed chunks → ChromaDB
│   └── config.py       # Filenames list and model pricing constants
├── documents/          # Place your PDF filings here (gitignored)
├── text_files/         # Auto-generated chunked text files (gitignored)
├── chroma_db/          # Persistent vector store (gitignored)
├── .env                # API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| Tesseract OCR | any recent (for unstructured) |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/stock_agent.git
cd stock_agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API keys

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
VOYAGE_API_KEY=your_voyageai_key_here
```

### 5. Add your SEC filings

Drop PDF files into the `documents/` folder and register their filenames in `src/config.py`:

```python
FILENAMES: list[str] = [
    "ionq.pdf",
    "rklb.pdf",
    "oklo.pdf",
]
```

---

## Running the Agent

```bash
cd src
python agent.py
```

On first run the agent will:
1. **Chunk** each PDF into structured text segments
2. **Embed** each segment with `voyage-3`
3. **Store** embeddings in a local ChromaDB database
4. **Start** the interactive CLI

Subsequent runs skip already-processed files and go straight to the CLI.

---

## Usage

```
Stock Agent started. Commands: 'quit' to exit, 'clear' to reset.

you: What does IonQ say about their revenue growth strategy?
claude: IonQ highlights ... [answer from filings]

you: What was Rocket Lab's gross profit in Q4 2024?
claude: According to the filing, Rocket Lab reported ...

you: clear
Conversation cleared.

you: quit

--- Session summary ---
Input tokens : 4,821
Output tokens: 312
Estimated cost: $0.019263
```

---

## Architecture

```
User input
    │
    ▼
agent() loop  ──► Anthropic Messages API (claude-haiku-4-5)
    │                        │
    │              stop_reason == "tool_use"
    │                        │
    ├──► rag_search()  ──► ChromaDB query ──► VoyageAI embed
    │
    └──► calculate()   ──► eval(expression)
```

1. The user's message is sent to Claude with two tools available.
2. Claude decides whether to call a tool or answer directly.
3. Tool results are appended to the conversation and the model is called again.
4. The loop exits when Claude produces a final `end_turn` text response.

---

## Adding New Filings

1. Copy the PDF to `documents/`.
2. Add the filename to `FILENAMES` in `src/config.py`.
3. Re-run `python agent.py` — only the new file will be chunked and ingested.

---

## Cost Estimates

Pricing is based on **Claude Haiku 4.5** rates (May 2026):

| Token type | Price |
|---|---|
| Input | $3.00 / 1M tokens |
| Output | $15.00 / 1M tokens |

Update `HAIKU_INPUT_PRICE_PER_M` and `HAIKU_OUTPUT_PRICE_PER_M` in `config.py` if you switch models.

---

## License

MIT