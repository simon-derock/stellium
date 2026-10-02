# One small LLM call in front of the live console: does the message ask for information, or is it
# small talk? Small talk gets a reply from the page instead of three pipelines saying "Not found".
# The benchmarks call the pipelines directly and never pass through here.
from __future__ import annotations

from dataclasses import dataclass

from src.llm import make_session

_PROMPT = (
    "Label a message sent to a question-answering service over Olympic events and Wikipedia "
    "articles. ASK: it asks for or about any fact, on any topic. CHAT: a greeting, thanks, small "
    "talk, a question about the assistant, or nothing answerable. Reply with one word."
)


@dataclass(frozen=True)
class Intent:
    label: str
    tokens: int
    latency_ms: float


async def classify_intent(query: str, provider: str) -> Intent:
    try:
        async with make_session(provider) as llm:
            result = await llm.chat(
                [{"role": "system", "content": _PROMPT}, {"role": "user", "content": query[:500]}],
                max_tokens=3,
                temperature=0.0,
            )
    except Exception:
        # The gate must never cost a real question its answer: if it fails, answer it.
        return Intent("ask", 0, 0.0)
    label = "chat" if result.content.strip().upper().startswith("CHAT") else "ask"
    return Intent(label, result.input_tokens + result.output_tokens, result.latency_ms)
