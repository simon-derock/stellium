# STELLIUM API for a small free instance (Render, 512 MB): the corpus and its keyword index load at
# start-up; reranking is a Cohere API call, so no model weights ship in the image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY hackathon-resources/corpus ./hackathon-resources/corpus
COPY hackathon-resources/questions/eval_public.jsonl ./hackathon-resources/questions/
COPY docs/metrics ./docs/metrics
RUN uv sync --locked --no-dev

RUN useradd --create-home stellium && chown -R stellium /app
USER stellium
EXPOSE 8000
CMD ["sh", "-c", "uv run --no-sync uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
