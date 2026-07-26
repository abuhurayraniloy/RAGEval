"""A/B testing endpoint: run one question through two configurations
concurrently and compare results."""

import asyncio
import json
import logging
from fastapi import HTTPException, status
from pydantic import BaseModel

from src.services.ab_pipeline import run_configured_pipeline
from src.db import AsyncSessionLocal, ABTestResult

logger = logging.getLogger("uvicorn.error")


class PipelineConfig(BaseModel):
    model: str | None = None
    # chunk_size: int | None = None
    top_k: int | None = None
    reranking: bool | None = None


class ABTestRequest(BaseModel):
    question: str
    config_a: PipelineConfig
    config_b: PipelineConfig


async def run_ab_test(request: ABTestRequest):
    """Run a question through two configurations simultaneously and
    compare answers, latency, and cost.

    Args:
            request: ABTestRequest with question, config_a, config_b

    Returns:
            Dictionary with both results and a stored record id
    """
    try:
        config_a_dict = request.config_a.model_dump(exclude_none=True)
        config_b_dict = request.config_b.model_dump(exclude_none=True)

        result_a, result_b = await asyncio.gather(
            run_configured_pipeline(request.question, config_a_dict),
            run_configured_pipeline(request.question, config_b_dict),
        )

        async with AsyncSessionLocal() as session:
            record = ABTestResult(
                question=request.question,
                config_a=json.dumps(config_a_dict),
                config_b=json.dumps(config_b_dict),
                answer_a=result_a["answer"],
                answer_b=result_b["answer"],
                latency_a_ms=result_a["latency_ms"],
                latency_b_ms=result_b["latency_ms"],
                cost_a_usd=result_a["cost_usd"],
                cost_b_usd=result_b["cost_usd"],
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)

        return {
            "status": "success",
            "question": request.question,
            "id": record.id,
            "result_a": result_a,
            "result_b": result_b,
            "comparison": {
                "latency_diff_ms": result_a["latency_ms"] - result_b["latency_ms"],
                "cost_diff_usd": round(result_a["cost_usd"] - result_b["cost_usd"], 6),
            },
        }

    except Exception as e:
        logger.error(f"A/B test failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while running the A/B test.",
        )
