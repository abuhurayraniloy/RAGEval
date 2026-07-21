"""
evals/agent.py

A minimal, from-scratch agentic loop — no LangChain, no LangGraph.

Structure:
  1. A system prompt describing the agent's capabilities.
  2. Tool definitions in OpenAI function-calling format (LiteLLM translates
     these to whatever the underlying provider expects - Groq, in our case).
  3. A loop: call the LLM, check if it requested a tool call, run the tool
     locally, feed the result back as a new message, repeat until the LLM
     responds with plain text instead of a tool call.
  4. Once tool-calling is done, a final structured call (via instructor)
     forces the answer into a validated AgentResponse Pydantic model
     instead of returning a raw string.

Conversation memory: every message is persisted to PostgreSQL via
evals.memory. Each call to run_agent_turn loads the last 10 messages (plus
any rolling summary of older history) instead of starting from scratch,
so the agent remembers context across separate calls.

Model note: llama-3.3-70b-versatile was tried first but consistently
produced malformed tool calls (a known rough edge with native tool calling
on some Groq-hosted models). llama-3.1-8b-instant has proven more reliable
for structured tool calling, so that's the default here. A retry is still
kept as a safety net in case a malformed tool call slips through anyway.

Usage:
    python -m evals.agent --base-url http://localhost:8000 --api-key <key>
"""

import argparse
import ast
import asyncio
import datetime
import json
import operator
import re
import time
import uuid

import httpx
import instructor
from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import BadRequestError, RateLimitError
from pydantic import BaseModel, Field

from evals.memory import build_context, save_message, maybe_summarize

load_dotenv()

MODEL = "groq/llama-3.1-8b-instant"

# Patch LiteLLM's completion function with instructor. This gives us an
# optional response_model=... argument on chat.completions.create(), which
# forces the model's output into a validated Pydantic model instead of a
# raw string - retrying automatically if the model's output doesn't fit.
instructor_client = instructor.from_litellm(completion)


class AgentResponse(BaseModel):
	"""Structured final answer from the agent - typed, not a raw string."""

	answer: str = Field(description="The direct answer to the user's question.")
	confidence: float = Field(
		description="Confidence in this answer, from 0.0 (pure guess) to 1.0 (certain).",
		ge=0.0,
		le=1.0,
	)
	sources: list[str] = Field(
		default_factory=list,
		description=(
			"Any sources used to arrive at the answer (e.g. knowledge base "
			"excerpts, or 'calculation' for computed results). Empty list "
			"if answered from general knowledge with no tool use."
		),
	)
	reasoning: str = Field(
		description="A brief explanation of how the answer was reached."
	)


SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.

You can:
- search_knowledge_base: search an internal knowledge base of technical
  documentation for relevant information.
- calculate: evaluate a mathematical expression safely.
- get_current_date: return today's date.

Use tools whenever a question requires information or computation you
don't already have. You may call multiple tools in sequence if a question
requires it (e.g. searching for a fact, then doing math with it). Once you
have everything you need, answer the user directly in plain text - do not
call a tool if you already have enough information to answer.
"""

TOOLS = [
	{
		"type": "function",
		"function": {
			"name": "search_knowledge_base",
			"description": (
				"Search the internal RAGEval knowledge base for relevant "
				"documentation chunks. Use this for any question about "
				"how the system works, its architecture, or its features."
			),
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "The search query text.",
					},
					"top_k": {
						"type": "integer",
						"description": "Number of results to return (default 3).",
					},
				},
				"required": ["query"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "calculate",
			"description": (
				"Evaluate a simple arithmetic expression, e.g. '12 * (4 + 3)'. "
				"Supports +, -, *, /, **, and parentheses. Use this for any "
				"math instead of trying to compute it yourself."
			),
			"parameters": {
				"type": "object",
				"properties": {
					"expression": {
						"type": "string",
						"description": "The math expression to evaluate.",
					},
				},
				"required": ["expression"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "get_current_date",
			"description": "Return today's date in YYYY-MM-DD format.",
			"parameters": {
				"type": "object",
				"properties": {},
			},
		},
	},
]


# --- Tool implementations -----------------------------------------------

def tool_search_knowledge_base(
	base_url: str, api_key: str, query: str, top_k: int = 3
) -> str:
	"""Call the running /search endpoint and return results as a string."""
	try:
		resp = httpx.post(
			f"{base_url}/search",
			json={"query": query, "top_k": top_k},
			headers={"X-API-Key": api_key},
			timeout=30.0,
		)
		resp.raise_for_status()
		data = resp.json()
		results = data.get("results", [])
		if not results:
			return "No relevant results found."
		return "\n\n".join(
			f"[score={r['score']:.3f}] {r['text']}" for r in results
		)
	except httpx.HTTPError as e:
		return f"Error calling search endpoint: {e}"


# Only a safe, whitelisted set of arithmetic operators - no arbitrary eval.
_ALLOWED_OPERATORS = {
	ast.Add: operator.add,
	ast.Sub: operator.sub,
	ast.Mult: operator.mul,
	ast.Div: operator.truediv,
	ast.Pow: operator.pow,
	ast.USub: operator.neg,
	ast.UAdd: operator.pos,
}


def _safe_eval(node):
	"""Recursively evaluate a math AST node, allowing only arithmetic ops."""
	if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
		return node.value
	if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
		return _ALLOWED_OPERATORS[type(node.op)](
			_safe_eval(node.left), _safe_eval(node.right)
		)
	if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
		return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
	raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def tool_calculate(expression: str) -> str:
	"""Safely evaluate a math expression using an AST whitelist - never
	Python's raw eval(), which would allow arbitrary code execution."""
	try:
		tree = ast.parse(expression, mode="eval")
		result = _safe_eval(tree.body)
		return str(result)
	except Exception as e:
		return f"Error evaluating expression: {e}"


def tool_get_current_date() -> str:
	"""Return today's date."""
	return datetime.date.today().isoformat()


def dispatch_tool(name: str, args: dict, base_url: str, api_key: str) -> str:
	"""Route a tool call by name to its implementation. Tolerant of minor
	type mismatches (e.g. a model sending "3" instead of 3 for an integer
	argument), since models don't always strictly respect the schema."""
	if name == "search_knowledge_base":
		top_k = args.get("top_k", 3)
		top_k = int(top_k) if top_k is not None else 3
		return tool_search_knowledge_base(base_url, api_key, args["query"], top_k)
	if name == "calculate":
		return tool_calculate(args["expression"])
	if name == "get_current_date":
		return tool_get_current_date()
	return f"Unknown tool: {name}"


def _extract_retry_seconds(error_message: str, default: float = 6.0) -> float:
	"""Pull a suggested wait time out of Groq's rate limit error message,
	e.g. "Please try again in 5.6s", falling back to a default if not found."""
	match = re.search(r"try again in ([\d.]+)s", error_message)
	if match:
		return float(match.group(1)) + 0.5  # small safety margin
	return default


def call_llm_with_retry(messages: list, max_retries: int = 5):
	"""Call the LLM with tool calling, retrying on:
	  - malformed tool-call errors (an occasional Groq native tool-calling
	    quirk), retried immediately, and
	  - rate limit errors (Groq's free-tier TPM cap), retried after waiting
	    the suggested backoff period from the error message.
	"""
	last_error = None
	for attempt in range(max_retries):
		try:
			return completion(
				model=MODEL,
				messages=messages,
				tools=TOOLS,
				tool_choice="auto",
			)
		except RateLimitError as e:
			last_error = e
			wait_seconds = _extract_retry_seconds(str(e))
			print(
				f"  [retry {attempt + 1}] Rate limit hit, waiting {wait_seconds:.1f}s..."
			)
			time.sleep(wait_seconds)
			continue
		except BadRequestError as e:
			last_error = e
			if "tool_use_failed" in str(e):
				print(f"  [retry {attempt + 1}] Model produced a malformed tool call, retrying...")
				continue
			raise
	raise last_error


# --- The agent loop, with persisted memory + structured final output -----

async def run_agent_turn(
	conversation_id: str, user_message: str, base_url: str, api_key: str,
	max_turns: int = 6, verbose: bool = True
) -> AgentResponse:
	"""Run one conversational turn with persisted memory: load context from
	PostgreSQL, run the tool-calling loop, then force the final answer
	through a validated AgentResponse via instructor before returning.

	Args:
		conversation_id: Identifier grouping this turn into a conversation
		user_message: The new user message for this turn
		base_url: Base URL of the running RAGEval API
		api_key: API key for the RAGEval API
		max_turns: Safety cap on tool-calling loop iterations
		verbose: Print each step as it happens

	Returns:
		A validated AgentResponse with answer, confidence, sources, and reasoning
	"""
	await save_message(conversation_id, "user", user_message)

	messages = await build_context(conversation_id, SYSTEM_PROMPT)

	for turn in range(max_turns):
		response = call_llm_with_retry(messages)
		message = response.choices[0].message

		if not message.tool_calls:
			# Tool-calling phase is done. Make one more call asking the
			# model to restate its answer in the required structured shape.
			messages.append({"role": "assistant", "content": message.content or ""})
			messages.append(
				{
					"role": "user",
					"content": (
						"Based on everything above, provide your final answer "
						"in the required structured format."
					),
				}
			)

			structured_answer = instructor_client.chat.completions.create(
				model=MODEL,
				messages=messages,
				response_model=AgentResponse,
			)

			if verbose:
				print(f"  [turn {turn + 1}] Final structured answer.")

			await save_message(conversation_id, "assistant", structured_answer.answer)
			await maybe_summarize(conversation_id)
			return structured_answer

		messages.append(message.model_dump())

		for tool_call in message.tool_calls:
			name = tool_call.function.name
			args = json.loads(tool_call.function.arguments)

			if verbose:
				print(f"  [turn {turn + 1}] Tool call: {name}({args})")

			result = dispatch_tool(name, args, base_url, api_key)

			if verbose:
				print(f"  [turn {turn + 1}] Tool result: {result[:200]}")

			messages.append(
				{"role": "tool", "tool_call_id": tool_call.id, "content": result}
			)
			await save_message(conversation_id, "tool", f"{name}: {result}")

	# Max turns reached without a clean finish - still return a typed
	# object, not a bare string, so callers never have to special-case this.
	return AgentResponse(
		answer="Max turns reached without a final answer.",
		confidence=0.0,
		sources=[],
		reasoning="The tool-calling loop exceeded its maximum allowed turns.",
	)


# --- Standalone test questions (no memory - each is a fresh conversation) --

TEST_QUESTIONS = [
	# 1. Single tool: search only
	"What database does RAGEval use for storing completions?",
	# 2. Single tool: calculate only
	"What is (128 * 4) + (256 / 8)?",
	# 3. Single tool: date only
	"What is today's date?",
	# 4. Two tools chained: search then calculate
	(
		"According to the knowledge base, how many candidates does the "
		"pipeline widen vector search to before reranking? Multiply that "
		"number by 3."
	),
	# 5. Two tools, unrelated: date and search in one question
	(
		"What is today's date, and separately, what chunking strategies "
		"does RAGEval support according to the knowledge base?"
	),
]


async def main():
	parser = argparse.ArgumentParser(description="Run the from-scratch agent")
	parser.add_argument("--base-url", default="http://localhost:8000")
	parser.add_argument("--api-key", required=True)
	args = parser.parse_args()

	for i, question in enumerate(TEST_QUESTIONS, start=1):
		print(f"\n{'=' * 70}")
		print(f"Q{i}: {question}")
		print(f"{'=' * 70}")

		# Each standalone test question is its own fresh conversation.
		conversation_id = str(uuid.uuid4())
		result = await run_agent_turn(conversation_id, question, args.base_url, args.api_key)

		print(f"\nANSWER:     {result.answer}")
		print(f"CONFIDENCE: {result.confidence}")
		print(f"SOURCES:    {result.sources}")
		print(f"REASONING:  {result.reasoning}")

		if i < len(TEST_QUESTIONS):
			time.sleep(3)  # brief pause between questions to ease TPM pressure


if __name__ == "__main__":
	asyncio.run(main())