# Resilient multi-provider LLM router with session-level model lock.
# Binding rule: same model everywhere within a single pipeline run.
# Provider and model stay fixed for each run; retries never mix models mid-run.
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from pathlib import Path
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
CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
CLOUDFLARE_OPENAI_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
)

# Fallback providers (run-level selection, never mid-run mixing)
GEMINI_MODEL = "gemini-3.8-flash"
MISTRAL_MODEL = "mistral-medium-latest"
COHERE_MODEL = os.environ.get("COHERE_CHAT_MODEL", "command-a-03-2025")
# Trial keys allow 20 calls per minute; production keys can set this to 0.
COHERE_CHAT_REQUEST_INTERVAL_S = float(os.environ.get("COHERE_CHAT_REQUEST_INTERVAL_S", "3.25"))
_COHERE_KEY_ENV_NAMES = ("COHERE_CHAT_API_KEY", "COHERE", "COHERE_BACKUP", "COHERE_KEY")
_MISTRAL_NAMED_KEY_SUFFIXES = (
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "SIXTEEN",
    "SEVENTEEN",
    "EIGHTEEN",
    "NINETEEN",
    "TWENTY",
    "TWENTYONE",
    "TWENTYTWO",
    "TWENTYTHREE",
)
_MISTRAL_KEY_ENV_NAMES = (
    *(f"KEY_{suffix}" for suffix in _MISTRAL_NAMED_KEY_SUFFIXES),
    "MISTRAL_API_KEY",
    *(f"MISTRAL_API_KEY_{i}" for i in range(1, 24)),
)
_cohere_key_pool_lock = asyncio.Lock()
_cohere_key_locks: dict[str, asyncio.Lock] = {}
_cohere_last_start_by_key: dict[str, float] = {}
_cohere_key_cursor = 0
_mistral_key_pool_lock = asyncio.Lock()
_mistral_key_cursor = 0


def _configured_cohere_api_keys() -> list[tuple[str, str]]:
    # Preserve configured key aliases while deduplicating identical secrets.
    configured: list[tuple[str, str]] = []
    seen: set[str] = set()
    for alias in _COHERE_KEY_ENV_NAMES:
        key = os.environ.get(alias, "").strip()
        if key and key not in seen:
            seen.add(key)
            configured.append((alias, key))
    return configured


async def _acquire_cohere_api_key() -> tuple[str, str]:
    # Round-robin distinct keys and enforce the trial request interval independently per key.
    global _cohere_key_cursor
    keys = _configured_cohere_api_keys()
    if not keys:
        raise RuntimeError("Cohere chat requires COHERE_CHAT_API_KEY or a configured COHERE key")

    async with _cohere_key_pool_lock:
        alias, key = keys[_cohere_key_cursor % len(keys)]
        _cohere_key_cursor += 1
        key_lock = _cohere_key_locks.setdefault(key, asyncio.Lock())

    async with key_lock:
        delay = COHERE_CHAT_REQUEST_INTERVAL_S - (
            time.monotonic() - _cohere_last_start_by_key.get(key, 0.0)
        )
        if delay > 0:
            await asyncio.sleep(delay)
        _cohere_last_start_by_key[key] = time.monotonic()
    return alias, key


def _configured_mistral_api_keys() -> list[tuple[str, str]]:
    # Preserve configured aliases while avoiding duplicate requests through copied keys.
    configured: list[tuple[str, str]] = []
    seen: set[str] = set()
    named_aliases = tuple(f"KEY_{suffix}" for suffix in _MISTRAL_NAMED_KEY_SUFFIXES)
    named_keys = [
        (alias, os.environ[alias].strip())
        for alias in named_aliases
        if os.environ.get(alias, "").strip()
    ]
    aliases = named_aliases if named_keys else _MISTRAL_KEY_ENV_NAMES[len(named_aliases) :]
    for alias in aliases:
        key = os.environ.get(alias, "").strip()
        if key and key not in seen:
            seen.add(key)
            configured.append((alias, key))
    return configured


def mistral_api_keys_configured() -> bool:
    return bool(_configured_mistral_api_keys())


async def _acquire_mistral_api_key() -> tuple[str, str]:
    # Rotate distinct configured keys while keeping model and prompt fixed per run.
    global _mistral_key_cursor
    keys = _configured_mistral_api_keys()
    if not keys:
        raise RuntimeError("Mistral chat requires KEY_ONE..KEY_TWENTYTHREE or MISTRAL_API_KEY")
    async with _mistral_key_pool_lock:
        selected = keys[_mistral_key_cursor % len(keys)]
        _mistral_key_cursor += 1
    return selected


class ProviderQuotaExceededError(RuntimeError):
    # Signals a provider quota response that cannot recover through short backoff retries.
    pass


class LLMCallResult:
    __slots__ = (
        "content",
        "input_tokens",
        "output_tokens",
        "model_name",
        "provider",
        "latency_ms",
        "credential_alias",
    )

    def __init__(
        self,
        content: str,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        provider: str,
        latency_ms: float,
        credential_alias: str = "",
    ) -> None:
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model_name = model_name
        self.provider = provider
        self.latency_ms = latency_ms
        self.credential_alias = credential_alias


class PersistentLLMResponseCache:
    # Append-only JSONL cache; each successful model response is fsynced before returning.
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._results: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as cache_file:
                for line_number, line in enumerate(cache_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Malformed LLM response cache at {self.path}:{line_number}"
                        ) from exc
                    cache_key = record.get("cache_key")
                    if isinstance(cache_key, str):
                        self._results[cache_key] = record

    def get(self, cache_key: str) -> LLMCallResult | None:
        with self._lock:
            record = self._results.get(cache_key)
            if record is None:
                return None
            return LLMCallResult(
                content=str(record["content"]),
                input_tokens=int(record["input_tokens"]),
                output_tokens=int(record["output_tokens"]),
                model_name=str(record["model_name"]),
                provider=str(record["provider"]),
                latency_ms=float(record["latency_ms"]),
                credential_alias=str(record.get("credential_alias", "")),
            )

    def put(self, cache_key: str, result: LLMCallResult) -> None:
        record = {
            "cache_key": cache_key,
            "content": result.content,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "model_name": result.model_name,
            "provider": result.provider,
            "latency_ms": result.latency_ms,
            "credential_alias": result.credential_alias,
        }
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as cache_file:
            cache_file.write(serialized)
            cache_file.flush()
            os.fsync(cache_file.fileno())
            self._results[cache_key] = record


_response_cache_registry_lock = threading.Lock()
_response_cache_registry: dict[str, PersistentLLMResponseCache] = {}


def _response_cache_from_env() -> PersistentLLMResponseCache | None:
    cache_path = os.environ.get("STELLIUM_LLM_RESPONSE_CACHE", "").strip()
    if not cache_path:
        return None
    resolved_path = str(Path(cache_path).expanduser().resolve())
    with _response_cache_registry_lock:
        cache = _response_cache_registry.get(resolved_path)
        if cache is None:
            cache = PersistentLLMResponseCache(Path(resolved_path))
            _response_cache_registry[resolved_path] = cache
        return cache


def _llm_response_cache_key(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    # Include the complete request contract so prompt/model changes cannot reuse stale output.
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        if _configured_mistral_api_keys():
            return await _call_mistral(MISTRAL_MODEL, messages, max_tokens, client)
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


# ---------------------------------------------------------------------------
# Gemini Provider (fallback)
# ---------------------------------------------------------------------------


async def _call_gemini(
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    client: httpx.AsyncClient | None = None,
) -> LLMCallResult:
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

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
        resp = await active_client.post(url, json=payload, headers=headers)
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
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    client: httpx.AsyncClient | None = None,
) -> LLMCallResult:
    credential_alias, api_key = await _acquire_mistral_api_key()
    url = "https://api.mistral.ai/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
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
    except httpx.HTTPStatusError as exc:
        setattr(exc, "credential_alias", credential_alias)
        raise
    finally:
        if own_client:
            await active_client.aclose()

    latency_ms = (time.perf_counter() - t0) * 1000
    return LLMCallResult(
        content=result_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_name=model,
        provider="mistral",
        latency_ms=latency_ms,
        credential_alias=credential_alias,
    )


# ---------------------------------------------------------------------------
# Cohere Chat API Provider
# ---------------------------------------------------------------------------


async def _call_cohere(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    client: httpx.AsyncClient | None = None,
) -> LLMCallResult:
    credential_alias, api_key = await _acquire_cohere_api_key()

    url = "https://api.cohere.com/v2/chat"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    active_client = client or httpx.AsyncClient(timeout=90.0)
    own_client = client is None
    try:
        resp = await active_client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", [])
        result_text = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if not result_text:
            raise RuntimeError("Cohere Chat API returned no text content")
        usage = data.get("usage", {})
        billed = usage.get("billed_units", {})
        tokens = usage.get("tokens", {})
        input_tokens = int(
            billed.get("input_tokens")
            or tokens.get("input_tokens")
            or max(1, sum(len(message.get("content", "")) // 4 for message in messages))
        )
        output_tokens = int(
            billed.get("output_tokens")
            or tokens.get("output_tokens")
            or max(1, len(result_text) // 4)
        )
    finally:
        if own_client:
            await active_client.aclose()

    return LLMCallResult(
        content=result_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_name=model,
        provider="cohere",
        latency_ms=(time.perf_counter() - t0) * 1000,
        credential_alias=credential_alias,
    )


# ---------------------------------------------------------------------------
# Session-Level Model Lock Router
# ---------------------------------------------------------------------------


class LockedLLMSession:
    # A single pipeline run uses ONE model throughout.
    # Retry transient throttling and gateway errors on the SAME model; do not retry exhausted quota.
    # Propagate provider failures after bounded retry handling.

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
        response_cache = _response_cache_from_env()
        cache_key = _llm_response_cache_key(
            self.provider, self.model, messages, max_tokens, temperature
        )
        if response_cache:
            cached_result = response_cache.get(cache_key)
            if cached_result:
                return cached_result

        last_exc: Exception | None = None
        attempt_limit = max_retries
        attempt = 0
        tried_mistral_aliases: set[str] = set()
        while attempt < attempt_limit:
            try:
                if self.provider == "cloudflare":
                    result = await _call_cloudflare(
                        self.model,
                        messages,
                        max_tokens,
                        temperature=temperature,
                        client=self._client,
                    )
                elif self.provider == "gemini":
                    result = await _call_gemini(messages, max_tokens, self._client)
                elif self.provider == "mistral":
                    result = await _call_mistral(self.model, messages, max_tokens, self._client)
                elif self.provider == "cohere":
                    result = await _call_cohere(
                        self.model,
                        messages,
                        max_tokens,
                        temperature=temperature,
                        client=self._client,
                    )
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                if response_cache:
                    response_cache.put(cache_key, result)
                return result
            except httpx.HTTPStatusError as e:
                if self.provider == "mistral":
                    credential_alias = getattr(e, "credential_alias", "")
                    if credential_alias:
                        tried_mistral_aliases.add(credential_alias)
                if self.provider == "cloudflare" and e.response.status_code == 429:
                    try:
                        payload = e.response.json()
                    except ValueError:
                        payload = {}
                    errors = payload.get("errors", []) if isinstance(payload, dict) else []
                    if not isinstance(errors, list):
                        errors = []
                    if any(
                        isinstance(error, dict) and error.get("code") == 4006 for error in errors
                    ):
                        raise ProviderQuotaExceededError(
                            "Cloudflare Workers AI daily neuron allocation is exhausted "
                            "(provider error 4006); retry after the quota resets."
                        ) from e
                # Transient rate limit or edge gateway saturation: retry on the SAME model.
                if e.response.status_code in (429, 500, 502, 503, 504, 524):
                    if self.provider == "mistral" and e.response.status_code == 429:
                        configured_keys = _configured_mistral_api_keys()
                        if len(tried_mistral_aliases) >= len(configured_keys):
                            last_exc = e
                            attempt += 1
                            break
                        # Move to another configured account immediately; avoid retrying a
                        # rate-limited account or waiting before trying a distinct credential.
                        attempt_limit = max(attempt_limit, len(configured_keys))
                        last_exc = e
                        attempt += 1
                        continue
                    retry_header = e.response.headers.get("retry-after")
                    wait = (
                        float(retry_header)
                        if retry_header and retry_header.isdigit()
                        else float(2**attempt)
                    )
                    await asyncio.sleep(min(wait, 30.0))
                    last_exc = e
                    attempt += 1
                    continue
                raise
        status_detail = (
            f" with HTTP status {last_exc.response.status_code}"
            if isinstance(last_exc, httpx.HTTPStatusError)
            else ""
        )
        raise RuntimeError(
            f"Provider {self.provider} failed after {attempt} attempts{status_detail}"
        ) from last_exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Uses the configured corpus-aligned provider in TigerGraph's 1024-dim vector space.
        from src.embeddings import (
            GRAPH_EMBEDDING_DIMENSION,
            embedding_client_from_env,
            is_graph_embedding_compatible,
        )

        embedding_client = embedding_client_from_env()
        if not embedding_client.is_configured:
            raise RuntimeError("A configured embedding API key is required for TigerGraph search")
        if not is_graph_embedding_compatible(embedding_client):
            raise RuntimeError(
                "Query embeddings must match the TigerGraph index model and dimension: "
                f"{GRAPH_EMBEDDING_DIMENSION}"
            )
        return embedding_client.embed_queries(texts)


def make_session(provider: str = "cloudflare") -> LockedLLMSession:
    # Factory for creating a locked session for one pipeline run.
    models = {
        "cloudflare": CLOUDFLARE_PRIMARY_MODEL,
        "gemini": GEMINI_MODEL,
        "mistral": os.environ.get("MISTRAL_CHAT_MODEL", MISTRAL_MODEL),
        "cohere": COHERE_MODEL,
    }
    model = models.get(provider, CLOUDFLARE_PRIMARY_MODEL)
    return LockedLLMSession(provider=provider, model=model)
