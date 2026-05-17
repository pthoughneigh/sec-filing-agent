# sec_filing_agent

A local CLI agent for querying SEC filings using a RAG pipeline powered by the Anthropic API, VoyageAI embeddings and reranking, and ChromaDB.

---

## Features

- **RAG search** — semantic search over chunked SEC filings with VoyageAI reranking
- **Calculator tool** — safe AST-based arithmetic on figures retrieved from filings
- **Senior analyst persona** — system prompt enforces citation requirements, fact/inference distinction, and no investment opinions
- **Multi-turn conversation** — full context preserved across turns in one session
- **Conversation export** — chat history saved to `conversations/` on exit
- **Idempotent ingestion** — re-running never re-embeds already-processed files
- **Token cost tracking** — session summary printed on exit

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
├── conversations/  # Exported chat histories (gitignored)
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
| Tesseract OCR | any recent (required by unstructured) |

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

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
VOYAGE_API_KEY=your_voyageai_key_here
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
1. **Chunk** each PDF into structured text segments grouped by section title
2. **Embed** each segment with `voyage-3` and store titles as metadata
3. **Store** embeddings in a local ChromaDB database
4. **Start** the interactive CLI

Subsequent runs skip already-processed files and go straight to the CLI.

---

## Usage

```
Stock Agent started. Commands: 'quit' to exit, 'clear' to reset.

you: What were IonQ's earnings in their most recent filing?
claude: [IonQ | 10-K | Revenue]
        IonQ reported revenue of $43.1M ...

you: What was Rocket Lab's gross profit margin?
claude: [Rocket Lab | 10-K | Gross Profit]
        The filing states gross profit was ...
        calculate: (gross_profit / revenue) * 100 = 14.2%

you: clear
Conversation cleared.

you: quit

--- Session summary ---
Input tokens : 6,340
Output tokens: 418
Estimated cost: $0.025290

Conversation saved to conversations/chat_0
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
    ├──► rag_search()  ──► ChromaDB query (n=10)
    │                           │
    │                      VoyageAI rerank-2 (top_k=3)
    │                           │
    │                      formatted context with citations
    │
    └──► calculate()   ──► safe AST evaluator
```

1. The user's message is sent to Claude with two tools available.
2. Claude decides whether to call a tool or answer directly.
3. Tool results are appended to the conversation and the model is called again.
4. The loop exits when Claude produces a final `end_turn` text response.
5. On exit, the conversation is saved to `conversations/`.

---

## Tuning Constants

Adjust these in `config.py` and `agent.py` to tune retrieval quality:

| Constant | Default | Description |
|---|---|---|
| `N_PARAMETERS` | 10 | Chunks fetched from ChromaDB before reranking |
| `TOP_K` | 3 | Chunks kept after reranking |
| `MAX_TOOL_TURNS` | 10 | Max tool calls per agent turn |
| `_CHUNK_SIZE_THRESHOLD` | 1000 | Max chars per chunk in chunker |

---

## Adding New Filings

1. Copy the PDF to `documents/`.
2. Add the filename to `FILENAMES` in `config.py`.
3. Re-run `python agent.py` — only the new file will be chunked and ingested.

---

## Cost Estimates

Pricing based on **Claude Haiku 4.5** rates (May 2026):

| Token type | Price |
|---|---|
| Input | $3.00 / 1M tokens |
| Output | $15.00 / 1M tokens |

Update `HAIKU_INPUT_PRICE_PER_M` and `HAIKU_OUTPUT_PRICE_PER_M` in `config.py` if you switch models.

---

## License

MIT