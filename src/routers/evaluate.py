"""RAG evaluation endpoint."""

import asyncio
import time
import logging
from fastapi import HTTPException, status
from pydantic import BaseModel

from src.services.rag_pipeline import rag_pipeline
from src.services.judge import judge_answer, JUDGE_MODEL

logger = logging.getLogger("uvicorn.error")

EVAL_CONCURRENCY = 30


class EvalQuestion(BaseModel):
    question: str
    expected: str


class EvalRequest(BaseModel):
    questions: list[EvalQuestion]


async def _evaluate_one(item: EvalQuestion, semaphore: asyncio.Semaphore) -> dict:
    """Evaluate a single question-answer pair.
    
    Args:
        item: Question and expected answer
        semaphore: Concurrency control semaphore
        
    Returns:
        Evaluation result with score and metrics
    """
    async with semaphore:
        logger.info(f"[evaluate] Running: {item.question[:60]}...")
        try:
            # Run the full RAG pipeline (no cache)
            rag = await rag_pipeline(item.question, use_cache=False)

            score, judge_cost = await judge_answer(
                item.question, item.expected, rag["answer"]
            )

            return {
                "question": item.question,
                "expected": item.expected,
                "actual": rag["answer"],
                "score": score,
                "latency_ms": rag["latency_ms"],
                "sources_used": len(rag["sources"]),
                "llm_cost_usd": rag.get("llm_cost_usd", 0.0),
                "judge_cost_usd": judge_cost,
            }

        except Exception as e:
            logger.error(
                f"[evaluate] Failed on '{item.question[:60]}': {e}", exc_info=True
            )
            return {
                "question": item.question,
                "expected": item.expected,
                "actual": "",
                "score": 0,
                "latency_ms": 0,
                "sources_used": 0,
                "llm_cost_usd": 0.0,
                "judge_cost_usd": 0.0,
                "error": str(e),
            }


async def evaluate(request: EvalRequest):
    """Run evaluation on a batch of questions.
    
    Args:
        request: EvalRequest with list of questions and expected answers
        
    Returns:
        Dictionary with accuracy, latency, costs, and per-question results
    """
    if not request.questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one question.",
        )

    try:
        batch_start = time.time()
        semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
        results = await asyncio.gather(
            *[_evaluate_one(item, semaphore) for item in request.questions]
        )
        batch_wall_time_ms = int((time.time() - batch_start) * 1000)

        total_latency_ms = sum(r["latency_ms"] for r in results)
        total_llm_cost = sum(r.get("llm_cost_usd", 0.0) for r in results)
        total_judge_cost = sum(r["judge_cost_usd"] for r in results)

        n = len(results)
        passed = sum(r["score"] for r in results)
        failed = n - passed

        return {
            "accuracy": round(passed / n, 3),
            "average_latency_ms": round(total_latency_ms / n, 1),
            "batch_wall_time_ms": batch_wall_time_ms,
            "eval_concurrency": EVAL_CONCURRENCY,
            "total_cost_usd": round(total_llm_cost + total_judge_cost, 6),
            "cost_breakdown": {
                "llm_cost_usd": round(total_llm_cost, 6),
                "judge_cost_usd": round(total_judge_cost, 6),
            },
            "total_questions": n,
            "passed": passed,
            "failed": failed,
            "judge_model": JUDGE_MODEL,
            "results": results,
        }

    except Exception as e:
        logger.error(f"[evaluate] Batch evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation batch failed: {str(e)}",
        )
