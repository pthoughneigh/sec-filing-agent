"""
agent.py

Agentic loop for the stock_agent CLI. Orchestrates tool use (RAG search and
calculator) via the Anthropic Messages API and drives a multi-turn conversation
in the terminal.
"""

import ast
import logging
import operator
import sys
import re
from pathlib import Path

import anthropic
from anthropic import APIConnectionError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

from chunker import load_pdfs
from config import (FILENAMES, HAIKU_INPUT_PRICE_PER_M, 
                   HAIKU_OUTPUT_PRICE_PER_M, SYSTEM_PROMPT, 
                   CHAT_OUTPUT_FOLDER, N_PARAMETERS, 
                   MAX_TOTAL_TURNS, INDEX_OF_LAST_SAVED_MESSAGE,
                   MAX_TOOL_TURNS)

from ingest import collection, ingest_all, model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

load_dotenv()

if not Path(".env").exists():
    log.warning("No .env file found — copy .env.example and fill in your API keys.")

try:
    client = anthropic.Anthropic(timeout=60.0)
    log.info("Anthropic client initialised successfully.")
except Exception as exc:
    log.error("Failed to initialise Anthropic client: %s", exc)
    log.error("Make sure ANTHROPIC_API_KEY is set in your .env file.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAFE_OPERATORS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
FILENAME_CACHE: list[str] = []


TOOLS: list[dict] = [
    {
        "name": "calculate",
        "description": (
            "Evaluates a mathematical expression derived from the user's question. "
            "Use this whenever the answer requires arithmetic, e.g. computing a "
            "revenue growth rate from two figures found in a filing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A valid arithmetic expression to evaluate, "
                        "e.g. '(150 - 100) / 100 * 100'. "
                        "Only numbers and basic operators (+, -, *, /, **, %) are supported."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "rag_search",
        "description": (
            "Retrieves relevant passages from the SEC-filing vector database to "
            "answer a question about a company. Use this before answering any "
            "question that requires factual information from the filings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The natural-language question to search for, "
                        "e.g. 'What does IonQ say about revenue growth strategy?'"
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "description": (
                        "How many document chunks to retrieve before reranking. Defaults to 15. "
                        "Increase for broad questions."
                    ),
                },
                "filename_filter": {
                    "type": "string",
                    "description": (
                        "Filename to restrict the search to a single filing, e.g. 'ionq.pdf'. "
                        "Use list_ingested_files to see valid filenames. Omit to search all filings."
                    )
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_ingested_files",
        "description": (
            "Tool used to return a list of used sources for chunking and ingestion."
            "Returns a numbered list of all filenames currently stored in the vector database."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        },

    },
    {
        "name": "compare_companies",
        "description": (
            "Retrieves the information about the each company mentioned in query for company comparison."
            "Use this tool when user wants to compare 2 or more companies, e.g. 'Compare me company X with Company Y' "
            "or 'Give me the difference between companies X, Y, Z'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                'query': {
                    "type": "string",
                    "description": (
                        "The natural-language question to search for, "
                        "e.g. 'Compare me company X with company Y.' or 'What is the difference between company X, Y and Z?'" 
                    )
                },
                'filenames': {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": ("Array of strings - filenames relevant to the query, used for comparison e.g. ['companyX.pdf', 'companyY.pdf']")
                }

            },
            "required": ['query', 'filenames']
        }
    }
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_source_filenames() -> set[str]:
    """
    Retrieve all unique source filenames from ChromaDB metadata.

    Queries the entire collection, extracts the 'source' field from each
    chunk's metadata, and returns a deduplicated set of filenames.
    Used internally to build the filename cache.

    Returns:
        set[str]: Unique source filenames (e.g. {'oklo_10k.pdf', 'oklo_8k.pdf'}).
                  Returns an empty set if the collection is empty or an error occurs.
    """
    try:
        results = collection.get()
        if not results['metadatas']:
            log.error("Collection is empty.")
            return set()
    except Exception as exc:
        log.error("Failed to retrieve source files: %s", exc)
        return set()

    sources = set()
    for metadata in results['metadatas']:
        source = metadata.get('source')
        if not source:
            log.warning("Chunk missing 'source' field, skipping: %s", metadata)
            continue
        try:
            sources.add(source.split('/')[2])
        except IndexError:
            log.warning("Unexpected source path format, skipping: %s", source)
            continue

    return sources

def _build_filename_cache() -> None:
    """
    Populate the global FILENAME_CACHE with available source filenames.

    Calls _get_source_filenames() and stores the result in FILENAME_CACHE
    as a list. Should be called once at startup before any tool that
    depends on the cache is invoked.

    Returns:
        None. Logs an error and returns early if no sources are found.
    """
    global FILENAME_CACHE
    sources = _get_source_filenames()
    if not sources:
        log.warning("No source filenames found in the vector store — FILENAME_CACHE will be empty.")
        return
    FILENAME_CACHE = list(sources)

def _write_to_file(conversation: list[str], file_number: int) -> None:
    """
    Write a list of plaintext conversation lines to a numbered chat file.

    Args:
        conversation: List of formatted strings, each representing one message.
        file_number: The numeric suffix for the output filename (e.g. 2 → chat_2).

    Raises:
        OSError: If the file cannot be opened or written to.
    """
    filepath = CHAT_OUTPUT_FOLDER / f'chat_{file_number}'
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for message in conversation:
                f.write(str(message) + '\n')
        log.info("Conversation saved to %s", filepath)
    except OSError as e:
        log.error("Failed to write conversation to %s: %s", filepath, e)
        raise

def _save_conversation(conversation: list[dict]) -> None:
    """
    Convert a conversation history to plaintext and save it to a numbered file
    in CHAT_OUTPUT_FOLDER. Tool result messages are excluded from the output.
    Each run creates a new file with an incremented number (chat_0, chat_1, ...).
    Does nothing if the conversation is empty.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys,
                      in the Anthropic API format.

    Raises:
        OSError: If the output folder cannot be created or the file cannot be written.
    """
    if not conversation:
        log.debug("Conversation is empty — skipping save.")
        return

    def get_assistant_text(content: list) -> str:
        """
        Extract plain text from an assistant content block list.
        Handles both raw dicts and Pydantic model objects (e.g. TextBlock).
        Returns an empty string if no text block is found (e.g. tool-only responses).

        Args:
            content: List of content blocks from an assistant message.

        Returns:
            The text of the first text block found, or an empty string.
        """
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                return block['text']
            if hasattr(block, 'text'):
                return block.text
        return ''

    try:
        CHAT_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Failed to create output folder %s: %s", CHAT_OUTPUT_FOLDER, e)
        raise

    plain_text: list[str] = [
        f"{x['role']}: {x['content']}\n" if x['role'] == 'user' and isinstance(x['content'], str)
        else f"{x['role']}: {get_assistant_text(x['content'])}\n"
        for x in conversation
        if not (x['role'] == 'user' and isinstance(x['content'], list))
        and not (x['role'] == 'assistant' and get_assistant_text(x['content']) == '')
    ]

    if not plain_text:
        log.debug("No printable messages after filtering — skipping save.")
        return

    try:
        files = sorted(CHAT_OUTPUT_FOLDER.iterdir())
        chat_files = [f for f in files if f.name.startswith('chat_')]

        if not chat_files:
            _write_to_file(plain_text, 0)
        else:
            match = re.search(r'\d+', chat_files[-1].name)
            last_file_number = int(match.group()) if match else 0
            _write_to_file(plain_text, last_file_number + 1)

    except OSError as e:
        log.error("Failed to read chat folder %s: %s", CHAT_OUTPUT_FOLDER, e)
        raise

def _check_collection_populated() -> None:
    """Warn the user if the vector store contains no documents.

    Called once at startup after ingestion so the user knows immediately if
    something went wrong during chunking/ingestion.
    """
    try:
        count = collection.count()
        if count == 0:
            log.warning(
                "The vector store is empty. Check that your PDFs are in ./documents/ "
                "and that chunking/ingestion completed without errors."
            )
        else:
            log.info("Vector store ready — %d chunks indexed.", count)
    except Exception as exc:
        log.warning("Could not verify collection size: %s", exc)


def _summarize_conversation(
        conversation: list[dict]
) -> list[dict]:
    """Summarize the conversation to reduce token usage in future turns.
    
    Calls the Claude API to produce a summary of ``conversation``, then returns
    a new list containing only the last 4 messages with the summary prepended
    as the first message. Falls back to returning the original conversation
    unchanged if any API error occurs.
    
    Args:
        conversation: The full conversation history as a list of role/content dicts.
    
    Returns:
        A new conversation list where older history is replaced by a single
        summary message of the form:
        ``{"role": "user", "content": "Summary of previous conversation: <text>"}``
        or the original conversation if an API error occurs.
    
    Raises:
        SystemExit: On unrecoverable API authentication errors (HTTP 401).
    """

    conversation = conversation.copy()

    summarize_system_prompt = (
        "You are a conversation summarizer. Given a conversation between a user and a financial assistant, "
        "produce a concise summary that preserves: the user's questions and intent, key numerical data "
        "(prices, percentages, figures) mentioned by the assistant, and important facts about any companies "
        "discussed. The summary will be used as context for continuing the conversation, so prioritize "
        "information the assistant would need to give consistent, accurate follow-up answers."
    )

    messages_to_summarize = conversation + [{"role": "user", "content": "Please summarize the conversation above."}]

    try:
        response = client.messages.create(
            model='claude-haiku-4-5',
            max_tokens=4096,
            system=summarize_system_prompt,
            messages=messages_to_summarize
        )
    except APITimeoutError:
        log.error("API request timed out after 60s.")
        return conversation
    except APIStatusError as exc:
        if exc.status_code == 401:
            log.error("Invalid Anthropic API key — check your .env file.")
            sys.exit(1)
        if exc.status_code == 429:
            log.warning("Rate limit hit (429).")
            return conversation
        log.error("Anthropic API error %d: %s", exc.status_code, exc.message)
        return conversation
    except APIConnectionError as exc:
        log.error("Could not reach the Anthropic API: %s", exc)
        return conversation

    log.debug("Response: %s", response)

    original = conversation.copy()
    conversation = conversation[INDEX_OF_LAST_SAVED_MESSAGE:]
    try:
        conversation.insert(0, {"role": "user", "content": f"Summary of previous conversation: {response.content[0].text}"})
    except IndexError as e:
        log.error("Response content is empty: %s", e)
        return original

    log.info("Summarization succeeded.")
    return conversation


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a parsed AST node using only safe arithmetic ops.

    Args:
        node: An AST node produced by ast.parse().

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: If the expression contains unsupported operations or types.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _SAFE_OPERATORS[op_type](left, right)  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](_eval_node(node.operand))  # type: ignore[operator]
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression using AST parsing.

    Only supports numeric literals and the operators +, -, *, /, //, %, **.
    No built-ins, imports, or arbitrary code can be executed.

    Args:
        expression: An arithmetic expression string, e.g. '(150 - 100) / 100 * 100'.

    Returns:
        The string representation of the result, or a descriptive error message
        the model can relay to the user.
    """
    log.debug("calculate() called with expression: %r", expression)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        log.debug("calculate() result: %s", result)
        return str(result)
    except ZeroDivisionError:
        log.warning("calculate() — division by zero in expression: %r", expression)
        return "Error: division by zero."
    except SyntaxError as exc:
        log.warning("calculate() — syntax error in expression %r: %s", expression, exc)
        return f"Error: invalid expression syntax — {exc}"
    except ValueError as exc:
        log.warning("calculate() — value error: %s", exc)
        return f"Error: {exc}"
    except Exception as exc:
        log.error("calculate() — unexpected error: %s", exc)
        return f"Error: could not evaluate expression — {exc}"


def list_ingested_files() -> str:
    """
    Query ChromaDB for all ingested document sources and return a formatted list.

    Retrieves metadata from the entire collection, extracts the 'source' field
    from each chunk, deduplicates by filename, and returns a numbered list
    as a string for Claude to read.

    Returns:
        str: A formatted string listing all unique source filenames,
             e.g. "This is the list of used sources:\n1. oklo_10k.pdf\n2. ..."
    """
    log.info("list_ingested_files() called")

    sources = _get_source_filenames()

    if not sources:
        return "Error: could not retrieve source files"
    
    message = "This is the list of used sources:"
    for i, source in enumerate(sources):
        message += f'\n{i+1}. {source}'

    return message


def rag_search(query: str, n_results: int = N_PARAMETERS, filename_filter: str | None = None) -> str:
    """Query the ChromaDB vector store and return formatted context chunks.

    Embeds ``query`` via all-mpnet-base-v2 model, retrieves the ``n_results`` nearest
    neighbours from ChromaDB, and returns the top-``TOP_K`` passages as a single formatted string.

    Args:
        query: Natural-language question to embed and search.
        n_results: Number of nearest-neighbour chunks to retrieve. 
        The final output contains at most ``TOP_K`` chunks.
        filename_filter: Name of a file for filtering the database search.

    Returns:
        A newline-joined string of context passages each prefixed with source
        metadata, or a descriptive error / warning string if any step fails.

    Raises:
        Does not propagate exceptions — all errors are caught, logged, and
        returned as descriptive strings so the caller always receives a ``str``.
    """
    log.info("rag_search() called — query: %r | n_results: %d", query, n_results)
    try:
        embeddings = model.encode([query]).tolist()
    except Exception as exc:
        log.error("Model embed failed: %s", exc)
        return f"Error: could not embed query — {exc}"

    try:
        n_results = min(n_results, collection.count())
    except Exception as exc:
        log.warning("Could not determine collection size: %s", exc)

    if n_results == 0:
        log.warning("rag_search() — collection is empty, no results returned.")
        return "No relevant passages found in the filings for this query."

    try:
        query_kwargs = {
            "query_embeddings": embeddings,
            "n_results": n_results,
        }
        if filename_filter:
            filename_filter = filename_filter.lower()+'.pdf' if not filename_filter.endswith('.pdf') else filename_filter
            source = ""
            for file in FILENAME_CACHE:
                if file.lower() == filename_filter.lower():
                    source = f"./documents/{file}"
            if not source:
                log.warning("rag_search() — filename_filter %r did not match any ingested file.", filename_filter)
                return f"No file matching '{filename_filter}' found in the database. Use list_ingested_files to see available sources."
            log.info("rag_search() — filtering by source: %s", filename_filter)
            query_kwargs["where"] = {"source": source}
            
        results = collection.query(
            **query_kwargs
        )
    except Exception as exc:
        log.error("ChromaDB query failed: %s", exc)
        return f"Error: vector store query failed — {exc}"

    documents: list[str] = results["documents"][0]
    metadatas: list[dict] = results["metadatas"][0]

    if not documents:
        log.warning("rag_search() — query returned no documents.")
        return "No relevant passages found in the filings for this query."

    context_docs = []
    for d, m in zip(documents, metadatas):
        log.debug("Retrieved chunk %d from %s", m['chunk_index'], m['source'])
        context_docs.append(
            f"[source: {m['source']}, section: {m.get('title', 'unknown')}]\n{d}"
        )

    log.info("rag_search() — returned %d chunks.", len(context_docs))
    return "\n".join(context_docs)

def compare_companies(query: str, filenames: list[str]) -> str:
    """Run a RAG search query against multiple files and aggregate the results.

    Args:
        query: The search query to run against each file.
        filenames: A list of filenames to search through individually.

    Returns:
        A single string with each file's results separated by a header and blank lines.

    Raises:
        ValueError: If query is empty or filenames list is empty.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not filenames:
        raise ValueError("filenames list must not be empty.")

    final_result = []
    for filename in filenames:
        if not filename or not filename.strip():
            log.warning("Skipping empty filename entry in filenames list.")
            continue
        result = rag_search(query, filename_filter=filename)
        result = f"=== Results for {filename} ===\n" + result + "\n"
        final_result.append(result)

    log.info("compare_companies: finished searching %d file(s) for query %r.", len(final_result), query)
    return "\n\n".join(final_result)

def run_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call by name and return its string result.

    Args:
        name: The tool name ('calculate', 'rag_search', 'list_ingested_files' or 'compare_companies').
        inputs: The tool input dict provided by the model.

    Returns:
        String result from the corresponding tool, or an error message if the
        tool name is not recognised.
    """
    log.info("run_tool() dispatching — tool: %r | inputs: %s", name, inputs)
    if name == "calculate":
        return calculate(inputs["expression"])
    if name == "rag_search":
        return rag_search(inputs["query"], int(inputs.get("n_results", N_PARAMETERS)), inputs.get('filename_filter', None))
    if name == 'list_ingested_files':
        return list_ingested_files()
    if name == 'compare_companies':
        return compare_companies(inputs['query'], filenames=inputs['filenames'])
    log.warning("run_tool() — unknown tool name: %r", name)
    return f"Error: unknown tool {name!r} — no action taken."


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


def agent(messages: list[dict]) -> tuple[str, int, int]:
    """Run the agentic loop until the model produces a final text response.

    The loop continues as long as the model requests tool calls, up to a
    maximum of MAX_TOOL_TURNS rounds to prevent infinite loops.

    Args:
        messages: Full conversation history in Anthropic message format.
                  Modified in-place during tool-use turns.

    Returns:
        A tuple of (answer_text, total_input_tokens, total_output_tokens).

    Raises:
        SystemExit: On unrecoverable API auth errors (401).
    """
    total_input = 0
    total_output = 0
    tool_turns = 0

    log.debug("agent() started — %d messages in history.", len(messages))

    while True:
        if tool_turns >= MAX_TOOL_TURNS:
            log.warning("Reached MAX_TOOL_TURNS (%d) — breaking loop.", MAX_TOOL_TURNS)
            return (
                "[error] The agent exceeded the maximum number of tool calls. "
                "Try rephrasing your question.",
                total_input,
                total_output,
            )

        try:
            with client.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=4096,
                system=[
                    {
                        'type': 'text',
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                tools=TOOLS,
                messages=messages,
            ) as stream:
                printed_prefix = False
                for event in stream:
                    if event.type == "text":
                        if not printed_prefix:
                             print("assistant: ", end="") 
                             printed_prefix = True
                        print(event.text, end="", flush=True)
                    elif event.type == "message_stop":
                        response = stream.get_final_message()
                        if printed_prefix:
                            print()

        except APITimeoutError:
            log.error("API request timed out after 60s.")
            return (
                "[error] Request timed out after 60 s. "
                "Try a shorter question or check your connection.",
                total_input,
                total_output,
            )
        except APIStatusError as exc:
            if exc.status_code == 401:
                log.error("Invalid Anthropic API key — check your .env file.")
                sys.exit(1)
            if exc.status_code == 429:
                log.warning("Rate limit hit (429).")
                return (
                    "[error] Rate limit reached. Wait a moment and try again.",
                    total_input,
                    total_output,
                )
            log.error("Anthropic API error %d: %s", exc.status_code, exc.message)
            return (
                f"[error] Anthropic API error {exc.status_code}: {exc.message}",
                total_input,
                total_output,
            )
        except APIConnectionError as exc:
            log.error("Could not reach the Anthropic API: %s", exc)
            return (
                f"[error] Could not reach the Anthropic API: {exc}",
                total_input,
                total_output,
            )

        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        log.debug(
            "API response — stop_reason: %s | tokens in/out: %d/%d",
            response.stop_reason,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        if response.stop_reason == "end_turn":
            text_block = next(
                (b for b in response.content if b.type == "text"), None
            )
            if text_block is None:
                log.warning("end_turn reached but no text block found in response.")
                return "(No text response generated)", total_input, total_output
            log.debug("agent() finished after %d tool turn(s).", tool_turns)
            return text_block.text, total_input, total_output

        if response.stop_reason == "tool_use":
            tool_turns += 1
            log.info("Tool use requested — turn %d/%d.", tool_turns, MAX_TOOL_TURNS)
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
        else:
            log.warning("Unexpected stop reason: %r", response.stop_reason)
            break

    return "(No response generated)", total_input, total_output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli() -> None:
    """Run an interactive terminal chat session with the stock analysis agent.

    Maintains full conversation history across turns. Prints token usage and
    estimated cost (Haiku pricing) on exit.

    Commands:
        quit  — End the session.
        clear — Reset conversation history.
    """
    conversation: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_turns = 0

    log.info("Stock Agent CLI started.")
    print("Stock Agent started. Commands: 'quit' to exit, 'clear' to reset.\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (KeyboardInterrupt, EOFError):
            log.info("Session interrupted by user.")
            print("\nExiting.")
            break

        if user_input.upper() == "QUIT":
            log.info("User issued QUIT — saving conversation.")
            _save_conversation(conversation)
            break
        if user_input.upper() == "CLEAR":
            log.info("User issued CLEAR — resetting conversation history.")
            conversation.clear()
            print("Conversation cleared.\n")
            continue
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})
        snapshot = len(conversation)
        
        answer, input_tokens, output_tokens = agent(conversation)
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        log.debug("Turn tokens — input: %d | output: %d", input_tokens, output_tokens)

        if answer.startswith("[error]"):
            log.warning("Error response received — rolling back conversation to snapshot %d.", snapshot)
            print(f"assistant: {answer}")
            del conversation[snapshot:]
        else:
            # Removes tool_use response from conversation
            conversation = conversation[:snapshot] 
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )

        total_turns += 1
        if total_turns >= MAX_TOTAL_TURNS:
            conversation = _summarize_conversation(conversation)
            total_turns = 0

    input_cost = (total_input_tokens / 1_000_000) * HAIKU_INPUT_PRICE_PER_M
    output_cost = (total_output_tokens / 1_000_000) * HAIKU_OUTPUT_PRICE_PER_M

    log.info(
        "Session ended — input tokens: %d | output tokens: %d | estimated cost: $%.6f",
        total_input_tokens,
        total_output_tokens,
        input_cost + output_cost,
    )

    print("\n--- Session summary ---")
    print(f"Input tokens : {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Estimated cost: ${input_cost + output_cost:.6f}")

if __name__ == "__main__":
    load_pdfs(FILENAMES)
    ingest_all(FILENAMES)
    _check_collection_populated()
    _build_filename_cache()
    cli()