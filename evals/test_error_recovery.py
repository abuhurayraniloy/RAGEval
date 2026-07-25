"""
evals/test_error_recovery.py

Intentionally triggers tool failures and structured-output failures to
verify the agent's error recovery: retrying with a different approach on
tool errors, retrying with a corrective prompt on invalid structured
output, and returning {"error": "Agent failed after 3 retries"} once
retries are exhausted - with every retry logged to PostgreSQL.

Usage:
    python -m evals.test_error_recovery --base-url http://localhost:8000 --api-key <key>
"""

import argparse
import asyncio
import uuid

from evals.agent import run_agent_turn

# Deliberately malformed inputs designed to trigger the calculator's
# error path (invalid syntax) and the search tool's error path (an
# unreachable/invalid base URL), so we can observe the recovery behavior.
TEST_CASES = [
    {
        "label": "Malformed math expression (should trigger AST rejection, then retry with a valid one)",
        "message": "Calculate this for me: 5 + + + * 3 ??? banana",
    },
    {
        "label": "Nonsensical search query (should still complete, possibly with 'no results')",
        "message": "Search the knowledge base for: zzz_nonexistent_term_qqq_12345",
    },
]


async def main():
    parser = argparse.ArgumentParser(description="Test agent error recovery")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--broken-base-url",
        default="http://localhost:9999",
        help="An intentionally invalid base URL to force search tool failures",
    )
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    print("=" * 70)
    print("TEST 1-2: Recoverable errors (valid base URL, bad inputs)")
    print("=" * 70)
    for case in TEST_CASES:
        conversation_id = str(uuid.uuid4())
        print(f"\n--- {case['label']} ---")
        result = await run_agent_turn(
            conversation_id, case["message"], args.base_url, args.api_key, verbose=True
        )
        if isinstance(result, dict):
            print(f"RESULT (error dict): {result}")
        else:
            print(f"ANSWER:     {result.answer}")
            print(f"CONFIDENCE: {result.confidence}")

    print("\n" + "=" * 70)
    print("TEST 3: Unrecoverable tool failure (search endpoint unreachable)")
    print("=" * 70)
    conversation_id = str(uuid.uuid4())
    result = await run_agent_turn(
        conversation_id,
        "Search the knowledge base for information about caching.",
        args.broken_base_url,  # intentionally wrong - every search call will fail
        args.api_key,
        verbose=True,
    )
    if isinstance(result, dict):
        print(f"RESULT (error dict): {result}")
    else:
        print(f"ANSWER:     {result.answer}")
        print(f"CONFIDENCE: {result.confidence}")
        print(f"SOURCES:    {result.sources}")

    print(
        "\nCheck the agent_retry_logs table in Postgres to confirm retries were logged:"
    )
    print(
        "  docker compose exec db psql -U user -d rageval_logs -c "
        '"SELECT retry_type, attempt_number, detail, created_at FROM agent_retry_logs '
        'ORDER BY created_at DESC LIMIT 20;"'
    )


if __name__ == "__main__":
    asyncio.run(main())
