# STELLIUM API for a small free instance (Render, 512 MB): the corpus index and the int8 reranker
# load at start-up, so the image carries both and the first question does not wait on a download.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HOME=/opt/hf

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY hackathon-resources/corpus ./hackathon-resources/corpus
COPY hackathon-resources/questions/eval_public.jsonl ./hackathon-resources/questions/
COPY docs/metrics ./docs/metrics
RUN uv sync --locked --no-dev

# Bake the pinned reranker into the image.
RUN uv run --no-sync python -c "from huggingface_hub import hf_hub_download as d; \
from src.coprocessor import _RERANKER_REPO as r, _RERANKER_FILE as f, _RERANKER_REVISION as v; \
d(repo_id=r, filename=f, revision=v); d(repo_id=r, filename='tokenizer.json', revision=v)"

RUN useradd --create-home stellium && chown -R stellium /app /opt/hf
USER stellium
EXPOSE 8000
CMD ["sh", "-c", "uv run --no-sync uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
