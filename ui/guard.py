"""Protection for a public deployment (Day 28): optional password and a daily run cap, so a shared link cannot
burn through your free-tier API quota. Pure functions, no Streamlit import, unit-tested."""
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path("data/usage.json")


def check_password(entered: str, expected: str) -> bool:
    """Constant-time comparison; an empty expected password means 'no password required'."""
    if not expected:
        return True
    return hmac.compare_digest(str(entered).encode(), str(expected).encode())


def max_runs_from_env(default: int = 10) -> int:
    try:
        return max(0, int(os.getenv("MAX_RUNS_PER_DAY", default)))
    except ValueError:
        return default


class RunLimiter:
    """Counts research runs per UTC day in a small JSON file. Not for real multi-user billing: it is a guard rail."""

    def __init__(self, path=None, max_per_day: int = None, today: str = None):
        self.path = Path(path or DEFAULT_PATH)
        self.max = max_runs_from_env() if max_per_day is None else max_per_day
        self.today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _used(self) -> int:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return int(data.get(self.today, 0))
        except (OSError, ValueError, AttributeError):
            return 0

    def remaining(self) -> int:
        return max(0, self.max - self._used())

    def allowed(self) -> bool:
        return self.remaining() > 0

    def record(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({self.today: self._used() + 1}), encoding="utf-8")   # old days are dropped
        except OSError:
            pass
