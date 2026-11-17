from bench.identity import index_identity, source_fingerprint


def test_index_identity_is_stable_and_ignores_cache_policy():
    base = dict(_target_="bench.index_factories.HNSWIndexFactory", M=32,
                ef_construction=100, seed=1, build_threads=4)
    first, digest_a = index_identity({**base, "cache_policy": "reuse"}, "db", "l2", "1.9.0", "src")
    _, digest_b = index_identity({**base, "cache_policy": "rebuild"}, "db", "l2", "1.9.0", "src")
    assert digest_a == digest_b
    assert "cache_policy" not in first["index"]


def test_index_identity_changes_with_the_construction_seed():
    base = dict(_target_="bench.index_factories.HNSWIndexFactory", M=32,
                ef_construction=100, build_threads=4, cache_policy="reuse")
    _, one = index_identity({**base, "seed": 1}, "db", "l2", "1.9.0", "src")
    _, two = index_identity({**base, "seed": 2}, "db", "l2", "1.9.0", "src")
    assert one != two


def test_source_fingerprint_follows_file_contents(tmp_path):
    (tmp_path / "faiss").mkdir()
    target = tmp_path / "faiss" / "a.cpp"
    target.write_text("one")
    before = source_fingerprint(tmp_path)
    target.write_text("two")
    assert source_fingerprint(tmp_path) != before
    (tmp_path / "faiss" / "notes.txt").write_text("ignored")
    assert source_fingerprint(tmp_path) == source_fingerprint(tmp_path)


def test_marker_is_valid_only_when_its_run_completed(tmp_path):
    from bench.identity import marker_valid
    marker = tmp_path / "run-x.txt"
    assert not marker_valid(marker)
    run = tmp_path / "out"
    marker.write_text(str(run))
    assert not marker_valid(marker)
    run.mkdir()
    (run / "complete.json").write_text('{"status": "complete"}')
    assert marker_valid(marker)
