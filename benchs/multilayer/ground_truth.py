from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import faiss
from bench.fingerprints import file_digest, vector_digest
from bench.groundtruth import (
    clusters_path, exact_neighbors, levels_path, load_s1, node_levels, s0_path, s1_path, upper_nodes,
    write_artifact, write_s1,
)
from bench.identity import index_identity, index_paths, source_fingerprint


def report(path: Path, created: bool) -> None:
    print(f"{'wrote' if created else 'present'}: {path}")


@hydra.main(version_base="1.3", config_path="configs", config_name="ground_truth")
def main(cfg: DictConfig) -> None:
    dataset = hydra.utils.instantiate(cfg.dataset)
    cache = Path(cfg.cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    database_id = vector_digest(dataset.database)
    query_id = vector_digest(dataset.queries)
    source_id = source_fingerprint(Path(__file__).resolve().parents[2])
    _, index_id = index_identity(
        OmegaConf.to_container(cfg.index, resolve=True), database_id, dataset.metric, faiss.__version__, source_id
    )
    index_path, _ = index_paths(cache, index_id)
    if not index_path.exists():
        raise FileNotFoundError(f"cached index not found: {index_path}")
    faiss.omp_set_num_threads(cfg.execution.search_threads)
    index_sha = file_digest(index_path)

    path = s0_path(cache, database_id, query_id, cfg.s0_k)
    if not path.exists():
        distances, ids = exact_neighbors(dataset.database, dataset.queries, cfg.s0_k, dataset.metric, cfg.batch_size)
        write_artifact(path, ids=ids, distances=distances)
        report(path, True)
    else:
        report(path, False)

    path = levels_path(cache, index_sha)
    if not path.exists():
        index = faiss.read_index(str(index_path))
        write_artifact(path, levels=node_levels(index), index_sha=np.array(index_sha))
        report(path, True)
    else:
        report(path, False)

    path = s1_path(cache, index_sha, cfg.s1_k)
    if not path.exists():
        with np.load(levels_path(cache, index_sha)) as stored:
            subset = upper_nodes(stored["levels"], level=1)
        distances, ids = exact_neighbors(
            dataset.database, dataset.queries, cfg.s1_k, dataset.metric, cfg.batch_size, subset=subset
        )
        write_s1(cache, index_sha, cfg.s1_k, index_sha, ids=ids, distances=distances, subset=subset)
        report(path, True)
    else:
        load_s1(cache, index_sha, cfg.s1_k, index_sha)
        report(path, False)

    if dataset.database_clusters is not None:
        path = clusters_path(cache, database_id, query_id)
        if not path.exists():
            write_artifact(
                path, centres=dataset.centres,
                database_clusters=dataset.database_clusters, query_clusters=dataset.query_clusters,
            )
            report(path, True)
        else:
            report(path, False)


if __name__ == "__main__":
    main()
