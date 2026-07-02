"""
tests/conftest.py

Session-wide fixtures shared across the entire test suite.
"""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_reranker():
    """
    Patches src.main.rerank for EVERY test in the suite, automatically,
    regardless of whether the individual test file remembers to mock it.

    WHY THIS EXISTS:
    Tests exercising /rag or /evaluate were passing locally (where the
    ms-marco-MiniLM-L-6-v2 model is already cached on disk from manual
    testing against a running server) but failing in CI (fresh runner, no
    cache, and HuggingFace rate-limits anonymous CI traffic with 429s -
    GitHub Actions runners share IP ranges, so this isn't rare). Any test
    that forgot to mock the reranker was one dependency-cache-state away
    from a real network call to huggingface.co and a flaky CI failure.

    This fixture makes that structurally impossible going forward: rerank()
    is patched at its call site in src.main to a deterministic pass-through
    that returns the first top_k candidates with a fixed score - no model
    load, no network call, ever, in any test.

    If a specific test wants to assert something about reranking behavior
    itself (e.g. score ordering), it can still layer its own
    patch("src.main.rerank", ...) inside that test - the two patches nest
    without conflict, and the more specific one just wins for that test.
    """
    with patch(
        "src.main.rerank",
        side_effect=lambda query, candidates, top_k=5: [
            (i, 1.0) for i in range(min(top_k, len(candidates)))
        ],
    ):
        yield
