"""Saved runs as JSON snapshots: save, list, open, overwrite, delete (Day 26, extended Day 27)."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

HISTORY_DIR = Path("data/history")
SCHEMA_VERSION = 2


def save_snapshot(snapshot: Dict, directory: Optional[Path] = None) -> Path:
    directory = Path(directory or HISTORY_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    overwrite_snapshot(path, snapshot)
    return path


def overwrite_snapshot(path, snapshot: Dict) -> None:
    Path(path).write_text(json.dumps({**snapshot, "schema_version": SCHEMA_VERSION}, indent=2, default=str),
                          encoding="utf-8")


def load_snapshot(path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def delete_snapshot(path, directory: Optional[Path] = None) -> bool:
    """Deletes one saved run. Refuses anything that is not a .json file directly inside the history folder."""
    directory = Path(directory or HISTORY_DIR).resolve()
    p = Path(path).resolve()
    if p.parent != directory or p.suffix != ".json" or not p.exists():
        return False
    p.unlink()
    return True


def list_snapshots(directory: Optional[Path] = None) -> List[Dict]:
    """Newest first. Unreadable files are skipped rather than breaking the app."""
    from ui.runstats import run_health   # local import keeps history importable on its own

    directory = Path(directory or HISTORY_DIR)
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.json"), reverse=True):
        try:
            d = load_snapshot(p)
        except (OSError, ValueError):
            continue
        results = d.get("fact_check_results") or []
        out.append({"path": p, "created_at": d.get("created_at", ""), "query": d.get("query", ""),
                    "claims": len(results), "state": run_health(d)["state"],
                    "confirmed": sum(1 for r in results if r.get("status") == "CONFIRMED")})
    return out
