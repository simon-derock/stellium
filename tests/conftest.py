# Test-wide isolation from developer state: no real .env, no real credentials, no shared caches.
import os
import re

import pytest

_CREDENTIAL_NAMES = re.compile(
    r"(COHERE.*|KEY_[A-Z]+|MISTRAL.*|GEMINI.*|CLOUDFLARE.*|JINA.*|EMBEDDING_KEY|TG_.*)"
)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # A developer's .env must never leak keys into tests: a test that silently finds a real key
    # can call a live API or skip the failure path it was written to check.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.evaluate.load_dotenv", lambda *args, **kwargs: False)
    for name in list(os.environ):
        if _CREDENTIAL_NAMES.fullmatch(name):
            monkeypatch.delenv(name)
    # Mocked embeddings must never land in the real cache that live runs read from.
    monkeypatch.setenv("STELLIUM_QUERY_EMBEDDING_CACHE", "off")
    monkeypatch.setenv("TG_USE_MOCK", "1")
    # Question allowances are tested on their own; elsewhere they would trip mid-suite.
    monkeypatch.setenv("STELLIUM_DAILY_QUESTION_CAP", "0")
    monkeypatch.setenv("STELLIUM_CLIENT_HOURLY_CAP", "0")
