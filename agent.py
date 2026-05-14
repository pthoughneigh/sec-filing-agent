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
from pathlib import Path

import anthropic
from anthropic import APIConnectionError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

from chunker import load_pdfs
from config import FILENAMES, HAIKU_INPUT_PRICE_PER_M, HAIKU_OUTPUT_PRICE_PER_M
from ingest import collection, ingest_all, vo

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
except Exception as exc:
    log.error("Failed to initialise Anthropic client: %s", exc)
    log.error("Make sure ANTHROPIC_API_KEY is set in your .env file.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of tool-call rounds per agent() invocation.
# Prevents runaway loops and unbounded API spend.
MAX_TOOL_TURNS: int = 10

# Allowed AST node types for the safe expression evaluator.
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
                    "type": "number",
                    "description": (
                        "How many document chunks to retrieve. Defaults to 2. "
                        "Increase to 4-5 for broad questions."
                    ),
                },
            },
            "required": ["query"],
        },
    },
]

SYSTEM_PROMPT = """You are a stock analysis assistant specialising in SEC filings.
Use `rag_search` to retrieve evidence from the filings before answering, and
`calculate` for any arithmetic. If the information is not in the filings, say so.
Be concise and factual — no filler text."""

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
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except SyntaxError as exc:
        return f"Error: invalid expression syntax — {exc}"
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: could not evaluate expression — {exc}"


def rag_search(query: str, n_results: int = 2) -> str:
    """Query the ChromaDB vector store and return formatted context chunks.

    Args:
        query: Natural-language question to embed and search.
        n_results: Number of nearest-neighbour chunks to return.

    Returns:
        A newline-joined string of context passages each prefixed with source
        metadata, or a descriptive error / warning string.
    """
    try:
        embeddings = vo.embed([query], model="voyage-3").embeddings
    except Exception as exc:
        log.error("VoyageAI embed failed: %s", exc)
        return f"Error: could not embed query — {exc}"

    # FIX 4: Clamp n_results to the actual collection size to avoid a
    # ChromaDB exception when the requested count exceeds available chunks.
    try:
        n_results = min(n_results, collection.count())
    except Exception as exc:
        log.warning("Could not determine collection size: %s", exc)

    if n_results == 0:
        return "No relevant passages found in the filings for this query."

    try:
        results = collection.query(
            query_embeddings=embeddings,
            n_results=n_results,
        )
    except Exception as exc:
        log.error("ChromaDB query failed: %s", exc)
        return f"Error: vector store query failed — {exc}"

    documents: list[str] = results["documents"][0]
    metadatas: list[dict] = results["metadatas"][0]

    if not documents:
        return "No relevant passages found in the filings for this query."

    context_docs = [
        f"[source: {m['source']}, chunk_index: {m['chunk_index']}]\n{d}"
        for d, m in zip(documents, metadatas)
    ]
    return "\n".join(context_docs)


def run_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call by name and return its string result.

    Args:
        name: The tool name ('calculate' or 'rag_search').
        inputs: The tool input dict provided by the model.

    Returns:
        String result from the corresponding tool, or an error message if the
        tool name is not recognised.
    """
    log.debug("Tool call: %s | inputs: %s", name, inputs)
    if name == "calculate":
        return calculate(inputs["expression"])
    if name == "rag_search":
        return rag_search(inputs["query"], int(inputs.get("n_results", 2)))
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
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except APITimeoutError:
            return (
                "[error] Request timed out after 60 s. "
                "Try a shorter question or check your connection.",
                total_input,
                total_output,
            )
        except APIStatusError as exc:
            if exc.status_code == 401:
                log.error("Invalid Anthropic API key. Check your .env file.")
                sys.exit(1)
            if exc.status_code == 429:
                return (
                    "[error] Rate limit reached. Wait a moment and try again.",
                    total_input,
                    total_output,
                )
            return (
                f"[error] Anthropic API error {exc.status_code}: {exc.message}",
                total_input,
                total_output,
            )
        except APIConnectionError as exc:
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
            # FIX 2: Find the first text block explicitly rather than blindly
            # taking [0], which may be a non-text block and would crash or
            # return the wrong content.
            text_block = next(
                (b for b in response.content if b.type == "text"), None
            )
            if text_block is None:
                return "(No text response generated)", total_input, total_output
            return text_block.text, total_input, total_output

        if response.stop_reason == "tool_use":
            tool_turns += 1
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
# Startup validation
# ---------------------------------------------------------------------------


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

    print("Stock Agent started. Commands: 'quit' to exit, 'clear' to reset.\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.upper() == "QUIT":
            break
        if user_input.upper() == "CLEAR":
            conversation.clear()
            print("Conversation cleared.\n")
            continue
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        # Snapshot the length AFTER appending the user message so that
        # del conversation[snapshot:] on error rolls back the user message
        # AND any intermediate tool-use turns agent() appended.
        # FIX 1: previously used pop() which only removed one item, leaving
        # tool-use turns in the history on error.
        snapshot = len(conversation)

        answer, input_tokens, output_tokens = agent(conversation)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        print(f"claude: {answer}\n")

        if answer.startswith("[error]"):
            # Roll back the user message AND any intermediate tool-use messages
            # that agent() appended to the conversation during tool-call rounds.
            del conversation[snapshot:]
        else:
            # FIX 3: Use a structured content list rather than a bare string to
            # keep the conversation history format consistent with tool-use turns
            # and avoid API errors on subsequent multi-turn requests.
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )

    input_cost = (total_input_tokens / 1_000_000) * HAIKU_INPUT_PRICE_PER_M
    output_cost = (total_output_tokens / 1_000_000) * HAIKU_OUTPUT_PRICE_PER_M

    print("\n--- Session summary ---")
    print(f"Input tokens : {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Estimated cost: ${input_cost + output_cost:.6f}")


if __name__ == "__main__":
    load_pdfs(FILENAMES)
    ingest_all(FILENAMES)
    _check_collection_populated()
    cli()