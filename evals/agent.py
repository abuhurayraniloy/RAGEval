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

Model note: llama-3.3-70b-versatile was tried first but consistently
produced malformed tool calls (a known rough edge with native tool calling
on some Groq-hosted models). llama-3.1-8b-instant has proven more reliable
for structured tool calling, so that's the default here. A retry is still
kept as a safety net in case a malformed tool call slips through anyway.

Usage:
    python evals/agent.py --base-url http://localhost:8000 --api-key <key>
"""

import argparse
import ast
import datetime
import json
import operator
import re
import time

import httpx
from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import BadRequestError, RateLimitError

load_dotenv()

MODEL = "groq/llama-3.1-8b-instant"

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


def call_llm_with_retry(messages: list, max_retries: int = 3):
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


# --- The agent loop ------------------------------------------------------

def run_agent(
	question: str, base_url: str, api_key: str, max_turns: int = 6, verbose: bool = True
) -> str:
	"""Run the agentic loop: call the LLM, execute any requested tool calls,
	feed results back, repeat until the LLM answers in plain text.

	Args:
		question: The user's question
		base_url: Base URL of the running RAGEval API (for search tool)
		api_key: API key for the RAGEval API (for search tool)
		max_turns: Safety cap on loop iterations to prevent infinite loops
		verbose: Print each step as it happens

	Returns:
		The agent's final plain-text answer
	"""
	messages = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": question},
	]

	for turn in range(max_turns):
		response = call_llm_with_retry(messages)
		message = response.choices[0].message

		# No tool call - the model is done, return its answer.
		if not message.tool_calls:
			if verbose:
				print(f"  [turn {turn + 1}] Final answer.")
			return message.content

		# Append the assistant's tool-call message to history before
		# appending tool results, matching the required message ordering.
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
				{
					"role": "tool",
					"tool_call_id": tool_call.id,
					"content": result,
				}
			)

	return "Max turns reached without a final answer."


# --- Test questions covering different tool combinations ------------------

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


def main():
	parser = argparse.ArgumentParser(description="Run the from-scratch agent")
	parser.add_argument("--base-url", default="http://localhost:8000")
	parser.add_argument("--api-key", required=True)
	args = parser.parse_args()

	for i, question in enumerate(TEST_QUESTIONS, start=1):
		print(f"\n{'=' * 70}")
		print(f"Q{i}: {question}")
		print(f"{'=' * 70}")
		answer = run_agent(question, args.base_url, args.api_key)
		print(f"\nANSWER: {answer}")

		if i < len(TEST_QUESTIONS):
			time.sleep(3)  # brief pause between questions to ease TPM pressure


if __name__ == "__main__":
	main()