import hashlib
import json
from pathlib import Path
from typing import Any

from bench.fingerprints import digest


def source_fingerprint(root: Path) -> str:
    source = hashlib.sha256()
    for path in sorted((root / "faiss").rglob("*")):
        if path.suffix in {".cpp", ".h", ".swig", ".py"}:
            source.update(str(path.relative_to(root)).encode())
            source.update(path.read_bytes())
    return source.hexdigest()


def index_identity(index_cfg: dict[str, Any], database_id: str, metric: str,
                   faiss_version: str, source_id: str) -> tuple[dict[str, Any], str]:
    identity = {
        "database": database_id,
        "metric": metric,
        "index": {key: value for key, value in index_cfg.items() if key != "cache_policy"},
        "faiss_version": faiss_version,
        "source": source_id,
    }
    return identity, digest(identity)


def index_paths(cache: Path, index_id: str) -> tuple[Path, Path]:
    return cache / f"index-{index_id}.faiss", cache / f"index-{index_id}.json"


def marker_valid(marker: Path) -> bool:
    if not marker.exists():
        return False
    done = Path(marker.read_text().strip()) / "complete.json"
    return done.exists() and json.loads(done.read_text()).get("status") == "complete"
