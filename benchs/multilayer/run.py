import argparse
import hashlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path

import hydra
import numpy as np
from bench.export import arrays, json_file, table
from bench.fingerprints import digest, file_digest, vector_digest
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_info, threadpool_limits

import faiss


def check(cfg: DictConfig) -> None:
    if not isinstance(cfg.execution.track, bool):
        raise TypeError("execution.track must be true or false")
    for name, value in {
        "k": cfg.k, "ground_truth.batch_size": cfg.ground_truth.batch_size,
        "execution.search_threads": cfg.execution.search_threads,
        "execution.blas_threads": cfg.execution.blas_threads,
        "execution.repetitions": cfg.execution.repetitions,
        "execution.query_batch_size": cfg.execution.query_batch_size,
        "diagnostics.batch_size": cfg.diagnostics.batch_size,
        "index.build_threads": cfg.index.build_threads,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.execution.warmups < 0:
        raise ValueError("warmups must be nonnegative")
    if cfg.diagnostics.query_limit is not None and cfg.diagnostics.query_limit < 1:
        raise ValueError("diagnostics.query_limit must be positive or null")
    if cfg.index.cache_policy not in {"reuse", "rebuild", "require-existing"}:
        raise ValueError("invalid index.cache_policy")


def run(cfg: DictConfig, output: Path) -> dict[str, float]:
    dataset = hydra.utils.instantiate(cfg.dataset)
    if cfg.k > len(dataset.database):
        raise ValueError("k exceeds database size")
    cache = Path(cfg.cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    database_id = vector_digest(dataset.database)
    query_id = vector_digest(dataset.queries)
    source = hashlib.sha256() # NOTE: source fingerprint invalidates caches after local C++ changes as well
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "faiss").rglob("*")):
        if path.suffix in {".cpp", ".h", ".swig", ".py"}:
            source.update(str(path.relative_to(root)).encode())
            source.update(path.read_bytes())
    source_id = source.hexdigest()
    identity = {
        "database": database_id, 
        "metric": dataset.metric,
        "index": OmegaConf.to_container(cfg.index, resolve=True),
        "faiss_version": faiss.__version__, 
        "source": source_id
    }
    identity["index"].pop("cache_policy")
    index_id = digest(identity)
    index_path = cache / f"index-{index_id}.faiss"
    reused = index_path.exists() and cfg.index.cache_policy != "rebuild"
    if not reused:
        if cfg.index.cache_policy == "require-existing":
            raise FileNotFoundError(f"cached index not found: {index_path}")
        factory = hydra.utils.instantiate(identity["index"])
        index = factory.build(dataset)
        temp = cache / f".{uuid.uuid4().hex}.faiss"
        faiss.write_index(index, str(temp))
        temp.replace(index_path)
        json_file(cache / f"index-{index_id}.json", identity)
    else:
        index = faiss.read_index(str(index_path))
    faiss.omp_set_num_threads(cfg.execution.search_threads)
    gt_id = digest(
        {
            "database": database_id, 
            "queries": query_id,
            "metric": dataset.metric, 
            "k": cfg.k,
            "faiss": faiss.__version__, 
            "source": source_id
        }
    )
    gt_path = cache / f"ground-truth-{gt_id}.npz"
    if not gt_path.exists():
        flat = (faiss.IndexFlatL2 if dataset.metric == "l2" else faiss.IndexFlatIP)(dataset.database.shape[1])
        flat.add(dataset.database)
        ds, ids = [], []
        for start in range(0, len(dataset.queries), cfg.ground_truth.batch_size):
            d, i = flat.search(dataset.queries[start:start + cfg.ground_truth.batch_size], cfg.k)
            ds.append(d)
            ids.append(i)
        temp = cache / f".{uuid.uuid4().hex}.npz"
        arrays(temp, distances=np.concatenate(ds), ids=np.concatenate(ids))
        temp.replace(gt_path)

    strategy = hydra.utils.instantiate(cfg.strategy)
    strategy.prepare(index)
    run_id = digest(
        {
            "index": index_id,
            "ground_truth": gt_id,
            "ef_search": strategy.ef_search,
            "effective_widths": strategy.effective_widths,
            "k": cfg.k
        }
    )
    marker = cache / f"run-{run_id}.txt"
    if marker.exists():
        json_file(output / "complete.json", {"schema_version": 1, "status": "duplicate", "of": marker.read_text()})
        return {}
    order = np.arange(len(dataset.queries), dtype=np.int64) # NOTE: timing and saved results use a fixed query order, recorded explicitly
    if cfg.execution.shuffle_queries:
        np.random.default_rng(cfg.execution.seed).shuffle(order)
    queries = np.ascontiguousarray(dataset.queries[order])
    batch_size = cfg.execution.query_batch_size
    batches = [(start, queries[start:start + batch_size]) for start in range(0, len(queries), batch_size)]
    for _ in range(cfg.execution.warmups):
        for _, batch in batches:
            strategy.search(index, batch, cfg.k)
    timings = []
    for trial in range(cfg.execution.repetitions):
        for start, batch in batches:
            t0 = time.perf_counter_ns()
            strategy.search(index, batch, cfg.k)
            elapsed = time.perf_counter_ns() - t0
            timings.append(
                {
                    "trial": trial, 
                    "query_start": start,
                    "query_count": len(batch),
                    "elapsed_ns": elapsed,
                    "search_threads": faiss.omp_get_max_threads()
                }
            )
    table(output / "timings.parquet", timings, cfg.output.compression)
    arrays(output / "query_ids.npz", query_ids=order)
    result_dir = output / "results"
    if cfg.output.save_neighbors or cfg.output.save_distances:
        result_dir.mkdir()
    with np.load(gt_path) as truth:
        exact_ids = truth["ids"]
    correct = correct_at_1 = 0
    for number, (start, batch) in enumerate(batches):
        distances, ids = strategy.search(index, batch, cfg.k)
        for query_index, neighbors in zip(order[start:start + len(batch)], ids):
            correct += np.intersect1d(neighbors[neighbors >= 0], exact_ids[query_index]).size
            correct_at_1 += int(neighbors[0] == exact_ids[query_index][0])
        if cfg.output.save_neighbors or cfg.output.save_distances:
            values = {"query_ids": order[start:start + len(batch)]}
            if cfg.output.save_neighbors:
                values["ids"] = ids
            if cfg.output.save_distances:
                values["distances"] = distances
            arrays(result_dir / f"batch-{number:06d}.npz", **values)

    elapsed_ns = sum(row["elapsed_ns"] for row in timings)
    query_count = sum(row["query_count"] for row in timings)
    metrics = {
        "recall_at_k": float(correct / (len(queries) * cfg.k)),
        "recall_at_1": float(correct_at_1 / len(queries)),
        "mean_query_ms": elapsed_ns / query_count / 1e6,
        "queries_per_second": query_count * 1e9 / elapsed_ns,
    }
    if cfg.diagnostics.enabled:
        diagnostic_dir = output / "diagnostics"
        diagnostic_dir.mkdir()
        limit = min(len(queries), cfg.diagnostics.query_limit or len(queries))
        totals = {}
        for number, start in enumerate(range(0, limit, cfg.diagnostics.batch_size)):
            end = min(start + cfg.diagnostics.batch_size, limit)
            values, aggregate = strategy.collect_diagnostics(index, queries[start:end], cfg.k, cfg.diagnostics.candidates)
            assert int(values["ndis"].sum()) == aggregate["ndis"], "per-level ndis does not match hnsw_stats.ndis"
            for name in ("ndis", "nhops"):
                if name in values:
                    totals[name] = totals.get(name, 0) + values[name].sum(axis=0)
            arrays(diagnostic_dir / f"batch-{number:06d}.npz", query_ids=order[start:end], **values)
            json_file(diagnostic_dir / f"batch-{number:06d}.json", aggregate)
        for name, counts in totals.items():
            metrics[f"{name}_per_query"] = float(counts.sum() / limit)
            for level, count in enumerate(counts):
                metrics[f"level_{level}/{name}_per_query"] = float(count / limit)

    metadata = {
        "schema_version": 1, "source_fingerprint": source_id,
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "faiss_version": faiss.__version__,
        "faiss_module": faiss.__file__, "faiss_compile_options": faiss.get_compile_options(),
        "native_libraries": threadpool_info(),
        "effective_search_threads": faiss.omp_get_max_threads(),
        "index_id": index_id, "index_path": str(index_path),
        "index_sha256": file_digest(index_path),
        "index_reused": reused, "ground_truth_path": str(gt_path),
        "database_id": database_id, "query_id": query_id,
        "database_shape": list(dataset.database.shape),
        "query_shape": list(dataset.queries.shape), "metric": dataset.metric,
        "distance_semantics": "squared_l2" if dataset.metric == "l2" else "inner_product",
        "strategy": strategy.name,
        "effective_widths": getattr(strategy, "effective_widths", None),
        "timing": "per-batch search call, includes Python binding overhead; excludes export",
    }
    json_file(output / "metadata.json", metadata)
    json_file(output / "metrics.json", metrics)
    json_file(output / "complete.json", {"schema_version": 1, "status": "complete"})
    marker.write_text(str(output))
    return metrics


def publish(cfg: DictConfig, output: Path, metrics: dict[str, float]) -> None:
    import wandb

    options = cfg.tracking
    tags = ([tag.strip() for tag in options.tags.split(",") if tag.strip()]
            if isinstance(options.tags, str) else list(options.tags))
    with wandb.init(
        project=options.project,
        entity=options.entity or None,
        name=HydraConfig.get().job.config_name,
        group="",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags,
        job_type="benchmark",
        mode=options.mode,
        dir=str(output),
        settings=wandb.Settings(disable_git=True, disable_code=True, save_code=False),
    ) as run:
        run.log(metrics)
        artifact = wandb.Artifact(f"results-{run.id}", type="benchmark-results")
        for name in ("config.yaml", "metadata.json", "metrics.json", "timings.parquet", "query_ids.npz", "complete.json"):
            artifact.add_file(str(output / name), name=name)
        for name in ("results", "diagnostics", ".hydra"):
            if (output / name).exists():
                artifact.add_dir(str(output / name), name=name)
        run.log_artifact(artifact)

        metadata = json.loads((output / "metadata.json").read_text())
        for enabled, field, kind, filename in (
            (options.upload_ground_truth, "ground_truth_path", "ground-truth", "ground_truth.npz"),
            (options.upload_index, "index_path", "hnsw-index", "index.faiss"),
        ):
            if enabled:
                path = Path(metadata[field])
                artifact = wandb.Artifact(path.stem, type=kind)
                artifact.add_file(str(path), name=filename)
                run.log_artifact(artifact)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check(cfg)
    output = Path(HydraConfig.get().runtime.output_dir)
    if (output / "metadata.json").exists() or (output / "complete.json").exists():
        raise FileExistsError(f"experiment artifacts already exist: {output}")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), output / "config.yaml")
    with threadpool_limits(limits=cfg.execution.blas_threads, user_api="blas"):
        metrics = run(cfg, output)
    if cfg.execution.track and metrics:
        publish(cfg, output, metrics)
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=None)
    arguments, overrides = parser.parse_known_args()
    if arguments.track is not None:
        overrides.append(f"execution.track={str(arguments.track).lower()}")
    sys.argv[1:] = overrides
    main()
