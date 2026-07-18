"""
evals/memory.py

Conversation memory for the agent: persists every message to PostgreSQL,
loads the last 10 messages per request, and summarizes older history via a
separate LLM call once the stored conversation exceeds a token budget.
"""

import tiktoken
from sqlalchemy import select, delete
from litellm import acompletion

from src.db import AsyncSessionLocal, ConversationMessage, ConversationSummary

TOKEN_BUDGET = 6000
RECENT_MESSAGE_COUNT = 10
SUMMARY_MODEL = "groq/llama-3.1-8b-instant"

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(texts: list[str]) -> int:
    """Count total tokens across a list of message contents."""
    return sum(len(_encoding.encode(t)) for t in texts if t)


async def save_message(conversation_id: str, role: str, content: str) -> None:
    """Persist a single conversation turn to PostgreSQL.

    Args:
        conversation_id: Identifier grouping messages into one conversation
        role: "user", "assistant", or "tool"
        content: The message text
    """
    async with AsyncSessionLocal() as session:
        session.add(
            ConversationMessage(
                conversation_id=conversation_id,
                role=role,
                content=content or "",
            )
        )
        await session.commit()


async def get_summary(conversation_id: str) -> str | None:
    """Fetch the current rolling summary for a conversation, if any."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
        row = result.scalar_one_or_none()
        return row.summary if row else None


async def get_recent_messages(
    conversation_id: str, limit: int = RECENT_MESSAGE_COUNT
) -> list[dict]:
    """Load the most recent N messages for a conversation, oldest first.

    Ordering DESC + LIMIT is required to actually get the *most recent* N
    rows (ASC + LIMIT would give the *oldest* N instead, since LIMIT always
    takes from the front of whatever order is specified). We then reverse
    the Python list to restore oldest-first order, which is what the LLM
    expects a conversation transcript to look like.

    Stored "tool" role messages are converted to "assistant" role text on
    reload. The strict tool_call_id linkage the API requires only exists
    within a single run_agent_turn's live in-memory message list - once a
    tool result is persisted and reloaded on a later, separate turn, that
    id can't be reconstructed into a still-valid tool-call/tool-result
    pair. Replaying it as plain narrative text keeps the historical
    context (what the tool found) without violating the API's structural
    requirements for "role: tool" messages.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
        )
        rows = list(result.scalars().all())
        rows.reverse()  # flip back to oldest-first for correct LLM message order

        messages = []
        for r in rows:
            if r.role == "tool":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"[Earlier tool result] {r.content}",
                    }
                )
            else:
                messages.append({"role": r.role, "content": r.content})
        return messages


async def get_all_messages(conversation_id: str) -> list[ConversationMessage]:
    """Load every stored message for a conversation, oldest first."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.asc())
        )
        return list(result.scalars().all())


async def build_context(conversation_id: str, system_prompt: str) -> list[dict]:
    """Build the message list to send to the LLM: system prompt (with any
    rolling summary folded in) plus the last N raw messages.

    Args:
        conversation_id: Identifier for this conversation
        system_prompt: The agent's base system prompt

    Returns:
        A list of {"role": ..., "content": ...} messages ready to send
    """
    summary = await get_summary(conversation_id)
    recent = await get_recent_messages(conversation_id)

    system_content = system_prompt
    if summary:
        system_content += f"\n\nSummary of earlier conversation (for context, not verbatim):\n{summary}"

    return [{"role": "system", "content": system_content}] + recent


async def maybe_summarize(conversation_id: str) -> None:
    """Check whether stored history exceeds the token budget, and if so,
    summarize everything except the most recent N messages into a single
    rolling summary, then delete the summarized rows so they aren't
    reloaded (or re-summarized) again.

    Args:
        conversation_id: Identifier for this conversation
    """
    all_messages = await get_all_messages(conversation_id)
    total_tokens = count_tokens([m.content for m in all_messages])

    if total_tokens <= TOKEN_BUDGET:
        return

    # Keep the most recent N messages raw; summarize everything older.
    to_summarize = (
        all_messages[:-RECENT_MESSAGE_COUNT]
        if len(all_messages) > RECENT_MESSAGE_COUNT
        else []
    )

    if not to_summarize:
        return

    existing_summary = await get_summary(conversation_id)

    transcript = "\n".join(f"{m.role}: {m.content}" for m in to_summarize)

    summarize_prompt = (
        "Summarize the following conversation history concisely, preserving "
        "any specific facts, names, numbers, or preferences the user "
        "mentioned - these details matter more than the general flow of "
        "the conversation. Write the summary in plain prose, a few "
        "sentences.\n\n"
    )
    if existing_summary:
        summarize_prompt += f"Existing summary so far:\n{existing_summary}\n\n"
    summarize_prompt += f"New messages to fold in:\n{transcript}"

    response = await acompletion(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": summarize_prompt}],
    )
    new_summary = response.choices[0].message.content

    async with AsyncSessionLocal() as session:
        # Upsert the summary row.
        existing = await session.execute(
            select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
        row = existing.scalar_one_or_none()
        if row:
            row.summary = new_summary
        else:
            session.add(
                ConversationSummary(
                    conversation_id=conversation_id, summary=new_summary
                )
            )

        # Delete the raw messages that were just folded into the summary,
        # so they don't get counted or re-summarized on the next check.
        summarized_ids = [m.id for m in to_summarize]
        await session.execute(
            delete(ConversationMessage).where(
                ConversationMessage.id.in_(summarized_ids)
            )
        )
        await session.commit()
