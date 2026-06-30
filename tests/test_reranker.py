"""
tests/test_reranker.py

Unit tests for src/reranker.py. The CrossEncoder model itself is mocked -
these tests verify our reranking logic (sorting, top_k truncation, index
mapping), not the model's actual relevance judgments, which can only be
evaluated against real data (see tests/eval_reranking.py).
"""

from unittest.mock import MagicMock, patch

import src.reranker as reranker_module
from src.reranker import rerank, RERANK_CANDIDATES_K


def _patched_model(scores):
    """Return a MagicMock CrossEncoder whose .predict() returns `scores`
    in order, regardless of input."""
    model = MagicMock()
    model.predict = MagicMock(return_value=scores)
    return model


class TestRerank:
    def setup_method(self):
        # Reset the lazy singleton before every test so mocks don't leak
        # across tests.
        reranker_module._model = None

    def test_empty_candidates_returns_empty_list(self):
        result = rerank("some query", [], top_k=5)
        assert result == []

    def test_reorders_by_score_descending(self):
        candidates = ["low relevance", "high relevance", "medium relevance"]
        scores = [0.1, 0.9, 0.5]

        with patch("src.reranker.CrossEncoder", return_value=_patched_model(scores)):
            result = rerank("query", candidates, top_k=3)

        # Expect index 1 (score .9) first, then index 2 (.5), then index 0 (.1)
        assert [idx for idx, _ in result] == [1, 2, 0]

    def test_respects_top_k_truncation(self):
        candidates = ["a", "b", "c", "d", "e"]
        scores = [0.5, 0.9, 0.1, 0.7, 0.3]

        with patch("src.reranker.CrossEncoder", return_value=_patched_model(scores)):
            result = rerank("query", candidates, top_k=2)

        assert len(result) == 2
        assert [idx for idx, _ in result] == [1, 3]  # scores .9 and .7

    def test_top_k_larger_than_candidates_returns_all(self):
        candidates = ["a", "b"]
        scores = [0.2, 0.8]

        with patch("src.reranker.CrossEncoder", return_value=_patched_model(scores)):
            result = rerank("query", candidates, top_k=10)

        assert len(result) == 2

    def test_returns_float_scores(self):
        candidates = ["a"]
        scores = [0.42]

        with patch("src.reranker.CrossEncoder", return_value=_patched_model(scores)):
            result = rerank("query", candidates, top_k=1)

        assert isinstance(result[0][1], float)
        assert result[0][1] == 0.42

    def test_model_called_with_query_candidate_pairs(self):
        candidates = ["doc one", "doc two"]
        scores = [0.5, 0.5]
        mock_model = _patched_model(scores)

        with patch("src.reranker.CrossEncoder", return_value=mock_model):
            rerank("my query", candidates, top_k=2)

        called_pairs = mock_model.predict.call_args[0][0]
        assert called_pairs == [("my query", "doc one"), ("my query", "doc two")]

    def test_model_loaded_only_once_across_multiple_calls(self):
        scores = [0.5]

        with patch(
            "src.reranker.CrossEncoder", return_value=_patched_model(scores)
        ) as mock_ce:
            rerank("q1", ["a"], top_k=1)
            rerank("q2", ["b"], top_k=1)
            rerank("q3", ["c"], top_k=1)

        mock_ce.assert_called_once()

    def test_candidate_k_constant_is_larger_than_typical_final_top_k(self):
        # Sanity check on the config: the candidate pool fetched from vector
        # search must be wider than the final reranked top_k (5 in /rag),
        # otherwise reranking has nothing extra to re-sort.
        assert RERANK_CANDIDATES_K > 5

    def test_ties_preserve_a_stable_relative_order(self):
        candidates = ["a", "b", "c"]
        scores = [0.5, 0.5, 0.5]

        with patch("src.reranker.CrossEncoder", return_value=_patched_model(scores)):
            result = rerank("query", candidates, top_k=3)

        # All tied - just confirm no candidates were dropped or duplicated
        assert sorted(idx for idx, _ in result) == [0, 1, 2]
