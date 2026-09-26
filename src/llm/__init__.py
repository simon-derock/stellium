# Resilient multi-provider LLM router with session-level model lock.
# Binding rule: same model everywhere within a single pipeline run.
# Fallback only triggers when a provider is completely unreachable (not on 429).
# On 429: exponential backoff on the SAME locked model.
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------

# Primary model: Cloudflare Workers AI llama-3.1-8b-instruct-fast
# ~9 neurons per call, native tool calling, 60k-128k context.
# 450 eval runs = ~4,000 neurons = 40% of 10k/day free tier.
CLOUDFLARE_PRIMARY_MODEL = (
    os.environ.get("CLOUDFLARE_MODEL") or "@cf/meta/llama-3.1-8b-instruct-fast"
)
CLOUDFLARE_EMBEDDING_MODEL = "@cf/baai/bge-m3"  # 1024-dim, 8k context
CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
CLOUDFLARE_OPENAI_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
)

# Fallback providers (run-level selection, never mid-run mixing)
GEMINI_MODEL = "gemini-2.0-flash"
MISTRAL_MODEL = "mistral-small-latest"


class LLMCallResult:
    __slots__ = ("content", "input_tokens", "output_tokens", "model_name", "provider", "latency_ms")

    def __init__(
        self,
        content: str,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        provider: str,
        latency_ms: float,
    ) -> None:
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model_name = model_name
        self.provider = provider
        self.latency_ms = latency_ms


# ---------------------------------------------------------------------------
# Cloudflare Workers AI Provider
# ---------------------------------------------------------------------------


async def _call_cloudflare(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    client: httpx.AsyncClient | None = None,
) -> LLMCallResult:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        # Fallback to Gemini if configured
        if os.environ.get("GEMINI_API_KEY"):
            return await _call_gemini(messages, max_tokens, client)
        # Fallback to Mistral if configured
        if os.environ.get("MISTRAL_API_KEY"):
            return await _call_mistral(messages, max_tokens, client)
        # Offline testing fallback
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return LLMCallResult(
            content=f"Not found in corpus (offline mode: {last_user[:50]})",
            input_tokens=max(1, len(last_user) // 4),
            output_tokens=15,
            model_name=model,
            provider="offline_mock",
            latency_ms=1.0,
        )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    active_client = client or httpx.AsyncClient(timeout=45.0)
    own_client = client is None

    result_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    try:
        # 1. Prefer OpenAI-compatible endpoint with exact ground-truth usage metadata
        openai_url = CLOUDFLARE_OPENAI_URL.format(account_id=account_id)
        openai_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = await active_client.post(openai_url, json=openai_payload, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            result_text = str(data["choices"][0]["message"]["content"])
            usage = data.get("usage", {})
            input_tokens = int(
                usage.get("prompt_tokens") or max(1, sum(len(m["content"]) // 4 for m in messages))
            )
            output_tokens = int(usage.get("completion_tokens") or max(1, len(result_text) // 4))
        else:
            # 2. Resilient fallback to Cloudflare native REST model endpoint
            native_url = CLOUDFLARE_BASE_URL.format(account_id=account_id) + f"/{model}"
            native_payload: dict[str, Any] = {
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": False,
                "temperature": temperature,
            }
            resp_native = await active_client.post(native_url, json=native_payload, headers=headers)
            resp_native.raise_for_status()
            native_data = resp_native.json()
            if not native_data.get("success", True) and "errors" in native_data:
                err_msg = native_data["errors"][0].get("message", "Cloudflare Workers AI Error")
                raise RuntimeError(f"Cloudflare Workers AI Error: {err_msg}")
            result_text = str(native_data["result"]["response"])
            prompt_text = " ".join(m["content"] for m in messages)
            input_tokens = max(1, len(prompt_text) // 4)
            output_tokens = max(1, len(result_text) // 4)
    finally:
        if own_client:
            await active_client.aclose()

    latency_ms = (time.perf_counter() - t0) * 1000
    return LLMCallResult(
        content=result_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_name=model,
        provider="cloudflare",
        latency_ms=latency_ms,
    )


async def _embed_cloudflare(
    texts: list[str],
    client: httpx.AsyncClient | None = None,
) -> list[list[float]]:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        # High-dimension normalized pseudo-semantic vector for offline testing
        import hashlib

        vectors: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(float(b) / 128.0 - 1.0) for b in (h * 32)[:1024]]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    url = CLOUDFLARE_BASE_URL.format(account_id=account_id) + f"/{CLOUDFLARE_EMBEDDING_MODEL}"
    payload = {"text": texts}
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    active_client = client or httpx.AsyncClient(timeout=30.0)
    own_client = client is None
    try:
        resp = await active_client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["result"]["data"]]
    finally:
        if own_client:
            await active_client.aclose()


# ---------------------------------------------------------------------------
# Gemini Provider (fallback)
# ---------------------------------------------------------------------------


async def _call_gemini(
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    client: httpx.AsyncClient | None = None,
) -> LLMCallResult:
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    # Convert OpenAI-style messages to Gemini format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] in ("user", "system") else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens},
    }

    t0 = time.perf_counter()
    active_client = client or httpx.AsyncClient(timeout=60.0)
    own_client = client is None
    try:
        resp = await active_client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        result_text: str = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get(
            "promptTokenCount", max(1, sum(len(m["content"]) // 4 for m in messages))
        )
        output_tokens = usage.get("candidatesTokenCount", max(1, len(result_text) // 4))
    finally:
        if own_client:
            await active_client.aclose()

    latency_ms = (time.perf_counter() - t0) * 1000
    return LLMCallResult(
        content=result_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_name=GEMINI_MODEL,
        provider="gemini",
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Mistral Provider (fallback)
# ---------------------------------------------------------------------------


async def _call_mistral(
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    client: httpx.AsyncClient | None = None,
) -> LLMCallResult:
    api_key = os.environ["MISTRAL_API_KEY"]
    url = "https://api.mistral.ai/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    t0 = time.perf_counter()
    active_client = client or httpx.AsyncClient(timeout=60.0)
    own_client = client is None
    try:
        resp = await active_client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        result_text: str = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get(
            "prompt_tokens", max(1, sum(len(m["content"]) // 4 for m in messages))
        )
        output_tokens = usage.get("completion_tokens", max(1, len(result_text) // 4))
    finally:
        if own_client:
            await active_client.aclose()

    latency_ms = (time.perf_counter() - t0) * 1000
    return LLMCallResult(
        content=result_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_name=MISTRAL_MODEL,
        provider="mistral",
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Session-Level Model Lock Router
# ---------------------------------------------------------------------------


class LockedLLMSession:
    # A single pipeline run uses ONE model throughout.
    # On 429: retry with exponential backoff on the SAME model.
    # On provider failure after max retries: raise RuntimeError (caller re-queues with fallback).

    def __init__(self, provider: str = "cloudflare", model: str = CLOUDFLARE_PRIMARY_MODEL) -> None:
        self.provider = provider
        self.model = model
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> LockedLLMSession:
        self._client = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> LLMCallResult:
        # All LLM calls within this session use the SAME locked model.
        # Retries with exponential backoff on transient errors (429, 500, 502, 503, 504, 524).
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                if self.provider == "cloudflare":
                    return await _call_cloudflare(
                        self.model,
                        messages,
                        max_tokens,
                        temperature=temperature,
                        client=self._client,
                    )
                elif self.provider == "gemini":
                    return await _call_gemini(messages, max_tokens, self._client)
                elif self.provider == "mistral":
                    return await _call_mistral(messages, max_tokens, self._client)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
            except httpx.HTTPStatusError as e:
                # Rate limit or edge gateway saturation: backoff and retry on SAME model
                if e.response.status_code in (429, 500, 502, 503, 504, 524):
                    retry_header = e.response.headers.get("retry-after")
                    wait = (
                        float(retry_header)
                        if retry_header and retry_header.isdigit()
                        else float(2**attempt)
                    )
                    await asyncio.sleep(min(wait, 30.0))
                    last_exc = e
                    continue
                raise
        raise RuntimeError(
            f"Provider {self.provider} failed after {max_retries} retries"
        ) from last_exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Uniform embedding model across all pipelines & agents using JinaEmbeddingClient.
        # Matches the 1024-dim HNSW vector index space in TigerGraph Savanna.
        from src.embeddings import JinaEmbeddingClient

        jina = JinaEmbeddingClient.from_env()
        if jina.is_configured:
            # Generate query embeddings with retrieval.query LoRA adapter
            return [jina.embed_query(t) for t in texts]

        # Secondary fallback if Cloudflare is explicitly configured
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
        if account_id and api_token:
            return await _embed_cloudflare(texts, self._client)

        # Deterministic offline fallback
        return await _embed_cloudflare(texts, self._client)


def make_session(provider: str = "cloudflare") -> LockedLLMSession:
    # Factory for creating a locked session for one pipeline run.
    models = {
        "cloudflare": CLOUDFLARE_PRIMARY_MODEL,
        "gemini": GEMINI_MODEL,
        "mistral": MISTRAL_MODEL,
    }
    model = models.get(provider, CLOUDFLARE_PRIMARY_MODEL)
    return LockedLLMSession(provider=provider, model=model)
