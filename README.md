# sec_filing_agent

A local CLI agent for querying SEC filings using a RAG pipeline powered by the Anthropic API, sentence-transformers local embeddings, and ChromaDB.

---

## Features

- **Streaming responses** — final answers stream token by token to the terminal via `client.messages.stream()`
- **RAG search** — semantic search over chunked SEC filings using local embeddings
- **Metadata filtering** — optionally restrict RAG search to a single filing by filename
- **Company comparison** — run the same query across multiple filings and aggregate results side by side
- **List ingested files tool** — Claude can query ChromaDB to list available sources
- **Filename cache** — source filenames loaded at startup for fast, case-insensitive filtering
- **Calculator tool** — safe AST-based arithmetic on figures retrieved from filings
- **Senior analyst persona** — system prompt enforces citation requirements, fact/inference distinction, and no investment opinions
- **Multi-turn conversation** — full context preserved across turns in one session
- **Conversation summarization** — older history summarized every 5 turns to reduce token usage
- **Conversation export** — chat history saved to `chats/` on exit
- **Idempotent ingestion** — re-running never re-embeds already-processed files
- **Token cost tracking** — session summary printed on exit
- **Fully local embeddings** — no embedding API key required; model downloads once and runs on your machine

---

## Project Structure

```
sec_filing_agent/
├── agent.py        # Agentic loop, tool dispatch, CLI entry point
├── chunker.py      # PDF → structured text chunks
├── ingest.py       # Embed chunks → ChromaDB
├── config.py       # Filenames, model pricing, and tuning constants
├── documents/      # Place your PDF filings here (gitignored)
├── text_files/     # Auto-generated chunked text files (gitignored)
├── chroma_db/      # Persistent vector store (gitignored)
├── chats/          # Exported chat histories (gitignored)
├── .env            # API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| Tesseract OCR | any recent (required by unstructured for hi_res PDF parsing) |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/sec_filing_agent.git
cd sec_filing_agent
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

Create a `.env` file in the project root. Only the Anthropic key is required — embeddings run locally:

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 5. Add your SEC filings

Drop PDF files into `documents/` and register their filenames in `config.py`:

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
python agent.py
```

On first run the agent will:
1. **Chunk** each PDF into structured text segments grouped by section title, using hi_res strategy with Tesseract OCR for image-based pages. Tables are saved as standalone chunks to preserve financial data.
2. **Embed** each segment with `all-mpnet-base-v2` (downloads automatically on first run) and store titles as metadata.
3. **Store** embeddings in a local ChromaDB database.
4. **Cache** ingested filenames for fast filtering at query time.
5. **Start** the interactive CLI.

Subsequent runs skip already-processed files and go straight to the CLI.

> **Note:** If you are switching from the VoyageAI branch, delete `chroma_db/` before running. The embedding dimensions are different (768 vs 1024) and the databases are incompatible.

---

## Usage

```
Stock Agent started. Commands: 'quit' to exit, 'clear' to reset.

you: What were IonQ's earnings in their most recent filing?
assistant: [IonQ | 10-K | Revenue]
           IonQ reported revenue of $43.1M ...

you: What was Rocket Lab's gross profit margin?
assistant: [Rocket Lab | 10-K | Gross Profit]
           The filing states gross profit was ...
           calculate: (gross_profit / revenue) * 100 = 14.2%

you: Compare the net losses of IonQ and Oklo.
assistant: === Results for ionq.pdf ===
           [IonQ | 10-K | Net Loss] ...
           === Results for oklo.pdf ===
           [Oklo | 10-K | Net Loss] ...

you: What files do you have?
assistant: This is the list of used sources:
           1. ionq.pdf
           2. rklb.pdf
           3. oklo.pdf

you: Search only Oklo's filing for their business description.
assistant: Oklo Inc. is developing advanced fission power plants ...

you: clear
Conversation cleared.

you: quit

--- Session summary ---
Input tokens : 6,340
Output tokens: 418
Estimated cost: $0.025290

Conversation saved to chats/chat_0
```

---

## Architecture

```
User input
    │
    ▼
agent() loop  ──► Anthropic Messages API (claude-haiku-4-5)
    │                        │
    │              stop_reason == "tool_use"     stop_reason == "end_turn"
    │                        │                           │
    ├──► rag_search()  ──► ChromaDB query (n=5)     stream chunks
    │         │                  │                  to terminal
    │   filename_filter?    all-mpnet-base-v2
    │   FILENAME_CACHE      local encode()
    │   lookup                   │
    │                   formatted context
    │                   with citations
    │
    ├──► compare_companies() ──► rag_search() per filename
    │                            aggregated results
    │
    ├──► calculate()   ──► safe AST evaluator
    │
    └──► list_ingested_files() ──► ChromaDB metadata scan
```

1. The user's message is sent to Claude with four tools available.
2. Claude decides whether to call a tool or answer directly.
3. Tool-use turns use `client.messages.stream()` — tool results are collected silently.
4. The final `end_turn` response streams token by token to the terminal.
5. Tool results are appended to the conversation and the model is called again.
6. On exit, the conversation is saved to `chats/`.

---

## Tools

| Tool | Description |
|---|---|
| `rag_search` | Semantic search over ingested filings. Supports optional `filename_filter` to restrict search to a single filing. |
| `compare_companies` | Runs the same query against multiple filings and returns results per file, for side-by-side comparison. |
| `calculate` | Safe AST-based arithmetic evaluator for figures from filings. |
| `list_ingested_files` | Returns a numbered list of all filenames currently in the vector store. |

---

## Tuning Constants

Adjust these in `config.py` to tune retrieval quality and conversation behaviour:

| Constant | Default | Description |
|---|---|---|
| `N_PARAMETERS` | 5 | Chunks fetched from ChromaDB per query |
| `MAX_TOOL_TURNS` | 10 | Max tool calls per agent turn |
| `MAX_TOTAL_TURNS` | 5 | Turns before conversation is summarized |
| `_CHUNK_SIZE_THRESHOLD` | 1000 | Max chars per chunk in chunker |

---

## Embedding Model

This branch uses [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) from sentence-transformers:

| Property | Value |
|---|---|
| Dimensions | 768 |
| Speed | Moderate |
| Quality | Best general-purpose sentence-transformers model |
| API key required | No |
| First-run download | ~420 MB (cached locally after that) |

---

## Adding New Filings

1. Copy the PDF to `documents/`.
2. Add the filename to `FILENAMES` in `config.py`.
3. Re-run `python agent.py` — only the new file will be chunked and ingested.

---

## Cost Estimates

Embeddings are free (local). Costs come from the Anthropic API only.

Pricing based on **Claude Haiku 4.5** rates (May 2026):

| Token type | Price |
|---|---|
| Input | $3.00 / 1M tokens |
| Output | $15.00 / 1M tokens |

Update `HAIKU_INPUT_PRICE_PER_M` and `HAIKU_OUTPUT_PRICE_PER_M` in `config.py` if you switch models.

---

## License

MIT