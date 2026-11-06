import argparse
import dataclasses
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

from bench.campaign import Cell, cells, order, overrides
from bench.identity import source_fingerprint

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_campaign(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def preflight(blocks: list[dict[str, Any]], root: Path) -> None:
    for block in blocks:
        for relative in block.get("requires", []):
            if not (root / relative).exists():
                raise FileNotFoundError(f"block {block['block']} needs {root / relative}")


def guard_fingerprint(ledger_dir: Path, fingerprint: str) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / "source_fingerprint.txt"
    if not path.exists():
        path.write_text(fingerprint)
    elif path.read_text() != fingerprint:
        raise RuntimeError(f"source fingerprint changed since this campaign started: {path}")


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.status: dict[tuple, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.status[self._key(row["cell"])] = row["status"]

    @staticmethod
    def _key(cell: Cell | dict[str, Any]) -> tuple:
        values = dataclasses.asdict(cell) if isinstance(cell, Cell) else cell
        return tuple(sorted(values.items()))

    def done(self, cell: Cell) -> bool:
        return self.status.get(self._key(cell), "failed") != "failed"

    def record(self, cell: Cell, status: str, seconds: float) -> None:
        self.status[self._key(cell)] = status
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps({"cell": dataclasses.asdict(cell), "status": status, "seconds": seconds}) + "\n")

    def counts(self) -> dict[str, int]:
        return dict(Counter(self.status.values()))


def subprocess_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def run_cell(cell: Cell, core: int, runner: Callable = subprocess_runner) -> str:
    argv = ["taskset", "-c", str(core), sys.executable, str(HERE / "run.py"),
            "--config-name", cell.config_name, *overrides(cell)]
    result = runner(argv)
    if result.returncode != 0:
        return "failed"
    for line in result.stdout.splitlines():
        if line.startswith("Artifacts: "):
            marker = Path(line.removeprefix("Artifacts: ").strip()) / "complete.json"
            if marker.exists():
                return json.loads(marker.read_text()).get("status", "failed")
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--core", type=int, required=True)
    parser.add_argument("--order-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    campaign = load_campaign(args.config)
    blocks = campaign["blocks"]
    sequence = order([cell for block in blocks for cell in cells(block)],
                     campaign["order_seed"] if args.order_seed is None else args.order_seed)
    if args.dry_run:
        for number, cell in enumerate(sequence, 1):
            print(f"{number:4d} {cell}")
        print(f"{len(sequence)} cells")
        return 0
    preflight(blocks, HERE)
    ledger_dir = HERE / "outputs" / campaign["name"]
    guard_fingerprint(ledger_dir, source_fingerprint(REPO))
    ledger = Ledger(ledger_dir / "campaign.jsonl")
    pending = [cell for cell in sequence if not ledger.done(cell)]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"skipped {len(sequence) - len([c for c in sequence if not ledger.done(c)])} recorded cells")
    durations = []
    for number, cell in enumerate(pending, 1):
        guard_fingerprint(ledger_dir, source_fingerprint(REPO))
        start = time.perf_counter()
        status = run_cell(cell, args.core)
        durations.append(time.perf_counter() - start)
        ledger.record(cell, status, durations[-1])
        eta = sum(durations) / len(durations) * (len(pending) - number)
        print(f"[{number}/{len(pending)}] {status} {cell} eta {eta / 60:.1f} min", flush=True)
    counts = Counter(ledger.status.values())
    print(f"complete={counts['complete']} duplicate={counts['duplicate']} failed={counts['failed']}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
