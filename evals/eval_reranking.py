"""
tests/eval_reranking.py

Compares retrieval BEFORE reranking (raw vector search order) vs AFTER
reranking (cross-encoder re-sorted order) on a fixed set of test queries.

This script talks to your *running* stack (POST /search for the "before"
view, POST /rag for the "after" view) rather than mocking anything, because
the whole point of this comparison is to see how the cross-encoder reorders
real chunks from your actual indexed corpus. It needs:

  1. The API running with chunks already indexed via /embed
     (docker compose up -d --build, then a handful of /embed calls).
  2. Network access to huggingface.co the first time it runs, so the
     ms-marco-MiniLM-L-6-v2 weights can download.

Usage:
    python tests/eval_reranking.py [--base-url http://localhost:8000]

Output:
    Prints a before/after ranking table per query, plus an aggregate summary
    (how often the #1 result changed, average rank movement, etc).

IMPORTANT - findings placeholder:
This file intentionally does NOT contain fabricated "results" in a comment.
Run it against your real corpus, paste me the output, and I'll fill in the
FINDINGS block at the bottom of this file with the actual numbers instead
of guessing at what a reranker "should" do on data I've never seen.
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import List

import httpx

# ── Edit this list to match your actual corpus / domain ──────────────────────
# 10 test queries is the minimum useful sample to eyeball reordering behavior;
# bump this up if you want statistically meaningful aggregate numbers.
TEST_QUERIES: List[str] = [
    "What is retrieval augmented generation?",
    "How does the caching layer work?",
    "What database is used for storing completions?",
    "How are embeddings generated?",
    "What happens when the cache is empty?",
    "How is the vector search collection configured?",
    "What chunking strategies are supported?",
    "How are errors handled in the embedding endpoint?",
    "What model is used for generating answers?",
    "How long are RAG results cached for?",
]


@dataclass
class QueryComparison:
    query: str
    before_ids: List[str] = field(default_factory=list)
    after_ids: List[str] = field(default_factory=list)
    before_texts: List[str] = field(default_factory=list)
    after_texts: List[str] = field(default_factory=list)


def fetch_before(
    client: httpx.Client, base_url: str, query: str, top_k: int = 5
) -> QueryComparison:
    """Raw vector search order, no reranking - hits /search directly."""
    resp = client.post(f"{base_url}/search", json={"query": query, "top_k": top_k})
    resp.raise_for_status()
    data = resp.json()
    comparison = QueryComparison(query=query)
    for r in data["results"]:
        comparison.before_ids.append(r["id"])
        comparison.before_texts.append(r["text"][:80])
    return comparison


def fetch_after(
    client: httpx.Client, base_url: str, comparison: QueryComparison
) -> None:
    """Reranked order - hits /rag, which internally widens the vector search
    to RERANK_CANDIDATE_K and reranks down to 5 with the cross-encoder."""
    resp = client.post(f"{base_url}/rag", json={"question": comparison.query})
    resp.raise_for_status()
    data = resp.json()
    for s in data["sources"]:
        comparison.after_ids.append(s["id"])
        comparison.after_texts.append(s["text"][:80])


def print_comparison(comparison: QueryComparison) -> None:
    print(f"\nQuery: {comparison.query}")
    print("-" * 100)
    print(f"{'#':<3}{'BEFORE (vector search)':<55}{'AFTER (reranked)':<55}")
    rows = max(len(comparison.before_texts), len(comparison.after_texts))
    for i in range(rows):
        before = comparison.before_texts[i] if i < len(comparison.before_texts) else ""
        after = comparison.after_texts[i] if i < len(comparison.after_texts) else ""
        print(f"{i+1:<3}{before:<55}{after:<55}")

    top1_changed = (
        comparison.before_ids[:1] != comparison.after_ids[:1]
        if comparison.before_ids and comparison.after_ids
        else None
    )
    print(f"Top-1 result changed: {top1_changed}")


def summarize(comparisons: List[QueryComparison]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total = len(comparisons)
    top1_changes = sum(
        1
        for c in comparisons
        if c.before_ids and c.after_ids and c.before_ids[0] != c.after_ids[0]
    )

    # Average rank displacement: for each id that appears in both lists,
    # how many positions did it move?
    displacements = []
    for c in comparisons:
        for after_rank, doc_id in enumerate(c.after_ids):
            if doc_id in c.before_ids:
                before_rank = c.before_ids.index(doc_id)
                displacements.append(abs(before_rank - after_rank))

    avg_displacement = sum(displacements) / len(displacements) if displacements else 0.0

    print(f"Queries evaluated:              {total}")
    print(f"Top-1 result changed:           {top1_changes}/{total}")
    print(f"Average rank displacement:      {avg_displacement:.2f} positions")
    print()
    print(
        "FINDINGS: <-- run this script against your live stack and paste me the\n"
        "output. I'll replace this placeholder with an honest writeup of what\n"
        "actually happened on your corpus, instead of guessing."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare vector search vs reranked retrieval"
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    comparisons = []
    with httpx.Client(timeout=60.0) as client:
        for query in TEST_QUERIES:
            try:
                comparison = fetch_before(client, args.base_url, query, args.top_k)
                fetch_after(client, args.base_url, comparison)
                print_comparison(comparison)
                comparisons.append(comparison)
            except httpx.HTTPError as e:
                print(f"Skipping query {query!r} due to error: {e}", file=sys.stderr)

    if comparisons:
        summarize(comparisons)
    else:
        print("No queries succeeded - is the API running and is the corpus indexed?")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# FINDINGS (fill in after running)
# ─────────────────────────────────────────────────────────────────────────────
#
# This section is intentionally empty. Run:
#
#     python tests/eval_reranking.py
#
# against your running stack with real indexed documents, then send me the
# output. I'll write the actual before/after observations here - e.g. how
# often the top result changed, whether reranking surfaced more relevant
# chunks that vector search had ranked lower, and any cases where reranking
# made things worse - grounded in what really happened on your data rather
# than a generic, possibly-wrong description of cross-encoder behavior.
