# Test-wide isolation from developer state on disk.
import pytest


@pytest.fixture(autouse=True)
def _no_shared_query_embedding_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mocked embeddings must never land in the real cache that live runs read from.
    monkeypatch.setenv("STELLIUM_QUERY_EMBEDDING_CACHE", "off")
