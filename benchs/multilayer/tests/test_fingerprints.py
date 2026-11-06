import json

from bench import fingerprints
from bench.fingerprints import recorded_file_digest


def _explode(path):
    raise AssertionError("the index file was hashed again")


def test_recorded_file_digest_reuses_the_sidecar(tmp_path, monkeypatch):
    target = tmp_path / "index.faiss"
    target.write_bytes(b"payload")
    sidecar = tmp_path / "index.json"
    sidecar.write_text("{}")
    first = recorded_file_digest(target, sidecar)
    assert json.loads(sidecar.read_text())["index_sha256"] == first
    monkeypatch.setattr(fingerprints, "file_digest", _explode)
    assert recorded_file_digest(target, sidecar) == first


def test_recorded_file_digest_rehashes_when_the_file_changed(tmp_path):
    target = tmp_path / "index.faiss"
    target.write_bytes(b"payload")
    sidecar = tmp_path / "index.json"
    sidecar.write_text("{}")
    first = recorded_file_digest(target, sidecar)
    target.write_bytes(b"different payload")
    assert recorded_file_digest(target, sidecar) != first
