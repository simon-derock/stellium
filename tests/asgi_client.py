# Synchronous adapter for HTTPX's async in-process ASGI transport.
import asyncio
from typing import Any

import httpx
from fastapi import FastAPI


class InProcessASGIClient:
    # Run app requests without Starlette TestClient's blocking portal.
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())
