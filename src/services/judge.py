"""LLM-based answer evaluation service."""

from litellm import acompletion, completion_cost
import logging

logger = logging.getLogger("uvicorn.error")

JUDGE_MODEL = "cerebras/gemma-4-31b"


async def judge_answer(question: str, expected: str, actual: str) -> tuple[int, float]:
    """Judge if an actual answer correctly addresses a question.
    
    Args:
        question: Original question
        expected: Expected reference answer
        actual: Actual answer to evaluate
        
    Returns:
        Tuple of (score, cost) where score is 1 (correct) or 0 (incorrect),
        and cost is the USD cost of the evaluation
    """
    prompt = (
        "You are a strict answer evaluator for a RAG system.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {actual}\n\n"
        "Does the actual answer correctly address the question with the same "
        "key information as the expected answer?\n\n"
        "Reply with ONLY the digit 1 (correct) or 0 (incorrect). "
        "No explanation, no punctuation — just the digit."
    )

    response = await acompletion(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
    )

    logger.info(response)

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError(f"Judge returned no content: {response}")

    raw = content.strip()

    score = next((int(c) for c in raw if c in "01"), 0)
    
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    return score, cost


def extract_cost(response) -> float:
    """Extract cost from a completion response.
    
    Args:
        response: Completion response object
        
    Returns:
        Cost in USD, or 0.0 if unable to calculate
    """
    try:
        return completion_cost(completion_response=response)
    except Exception:
        return 0.0
