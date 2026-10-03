# Question allowances for the public API. Every question spends model calls from a monthly
# allowance, so neither one visitor nor a script may spend it in an afternoon.
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request

_HOUR_S = 3600.0


def _cap(name: str, default: int) -> int:
    # Read on every question so an operator can change a cap without a redeploy; 0 turns it off.
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def client_id(request: Request) -> str:
    # Netlify's proxy names the visitor; Render forwards the chain. A caller that skips the proxy
    # can forge these headers, which only dodges the per-visitor cap: the daily cap still holds.
    for header in ("x-nf-client-connection-ip", "x-forwarded-for"):
        value = request.headers.get(header, "")
        if value:
            return value.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


class QuestionBudget:
    def __init__(self) -> None:
        self._day = ""
        self._spent = 0
        self._recent: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def spend(self, client: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        daily = _cap("STELLIUM_DAILY_QUESTION_CAP", 100)
        hourly = _cap("STELLIUM_CLIENT_HOURLY_CAP", 20)
        today = datetime.fromtimestamp(now, UTC)
        with self._lock:
            if today.date().isoformat() != self._day:
                self._day, self._spent = today.date().isoformat(), 0
                # Visitors not seen for an hour carry nothing forward.
                self._recent = {
                    k: v for k, v in self._recent.items() if v and v[-1] > now - _HOUR_S
                }
            if daily and self._spent >= daily:
                midnight = datetime.combine(
                    today.date() + timedelta(days=1), datetime.min.time(), UTC
                )
                raise HTTPException(
                    status_code=429,
                    detail="Today's live questions for this demo are used up. They reset at "
                    "00:00 UTC; the Benchmark and Docs pages work as usual.",
                    headers={"Retry-After": str(int(midnight.timestamp() - now) + 1)},
                )
            stamps = self._recent.setdefault(client, deque())
            while stamps and stamps[0] <= now - _HOUR_S:
                stamps.popleft()
            if hourly and len(stamps) >= hourly:
                wait_s = int(stamps[0] + _HOUR_S - now) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"This demo answers {hourly} questions an hour per visitor. "
                    f"Try again in {max(1, wait_s // 60)} min.",
                    headers={"Retry-After": str(wait_s)},
                )
            stamps.append(now)
            self._spent += 1
