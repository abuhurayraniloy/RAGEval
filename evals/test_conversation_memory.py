"""
evals/test_conversation_memory.py

Runs a 20-turn conversation to verify the agent retains early context
after enough turns have passed that raw history would normally have
fallen out of the last-10-messages window - proving the summarization
path actually works, not just the raw recent-history window.

Now that run_agent_turn returns a structured AgentResponse (answer,
confidence, sources, reasoning) instead of a raw string, this script
prints all four fields for each turn.

Usage:
    python -m evals.test_conversation_memory --base-url http://localhost:8000 --api-key <key>
"""

import argparse
import asyncio
import time
import uuid

from evals.agent import run_agent_turn

# Turn 1 plants a specific, checkable fact. Turns 2-19 are filler to push
# turn 1 well outside the last-10-messages raw window. Turn 20 asks the
# agent to recall the fact from turn 1 - this only works if either the
# raw window still covers it (it won't, by turn 20) or the summary
# correctly preserved it.
CONVERSATION = [
    "My favorite number is 42 and I'm working on a project called Aurora.",
    "What's the weather like in general during autumn?",
    "Can you calculate 15 * 3 for me?",
    "What is today's date?",
    "Tell me about hybrid search in general terms.",
    "What is 100 divided by 4?",
    "What chunking strategies exist in RAG systems generally?",
    "Calculate 7 * 8 for me.",
    "What's a cross-encoder used for?",
    "What is 9 squared?",
    "Explain what RRF fusion means.",
    "Calculate 50 - 13.",
    "What's the purpose of a reranker?",
    "What is 6 * 6?",
    "Tell me about vector databases in general.",
    "Calculate 200 / 5.",
    "What does BM25 stand for conceptually?",
    "What is 3 to the power of 4?",
    "One more calculation: 45 + 55.",
    "What's my favorite number, and what project am I working on?",
]


async def main():
    parser = argparse.ArgumentParser(description="Test 20-turn conversation memory")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    conversation_id = str(uuid.uuid4())
    print(f"Conversation ID: {conversation_id}\n")

    for i, message in enumerate(CONVERSATION, start=1):
        print(f"{'=' * 70}")
        print(f"Turn {i}: {message}")
        print(f"{'=' * 70}")

        result = await run_agent_turn(
            conversation_id, message, args.base_url, args.api_key, verbose=False
        )

        print(f"Agent:      {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Sources:    {result.sources}")
        print(f"Reasoning:  {result.reasoning}\n")

        if i < len(CONVERSATION):
            time.sleep(15)  # give Groq's TPM budget time to refill between turns

    print(f"\n{'=' * 70}")
    print("CHECK: Did the final answer correctly recall '42' and 'Aurora'?")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
