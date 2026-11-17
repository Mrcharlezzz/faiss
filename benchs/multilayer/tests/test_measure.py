import numpy as np

from bench.measure import score, timed_passes


class CountingStrategy:
    def __init__(self, ids):
        self.ids = ids
        self.calls = 0

    def search(self, index, queries, k):
        self.calls += 1
        return np.zeros(self.ids.shape, dtype="float32"), self.ids


def test_score_runs_the_requested_untimed_passes_and_scores_the_last():
    strategy = CountingStrategy(ids=np.array([[2, 1], [0, 3]]))
    batches = [(0, np.zeros((2, 2), dtype="float32"))]
    result = score(strategy, None, batches, 2, np.array([0, 1]),
                   exact_ids=np.array([[2, 9], [3, 9]]), passes=3)
    assert strategy.calls == 3
    assert result.recall_at_k == 0.5
    assert result.recall_at_1 == 0.5
    assert len(result.results) == 1


def test_score_always_runs_one_pass_when_warmups_is_zero():
    strategy = CountingStrategy(ids=np.array([[0]]))
    result = score(strategy, None, [(0, np.zeros((1, 2), dtype="float32"))], 1,
                   np.array([0]), exact_ids=np.array([[0]]), passes=0)
    assert strategy.calls == 1
    assert result.recall_at_1 == 1.0


def test_timed_passes_records_every_pass_and_batch():
    strategy = CountingStrategy(ids=np.zeros((2, 1), dtype="int64"))
    batches = [(0, np.zeros((2, 2), dtype="float32")), (2, np.zeros((2, 2), dtype="float32"))]
    rows = timed_passes(strategy, None, batches, 1, repetitions=3, search_threads=1)
    assert len(rows) == 6
    assert sorted({row["trial"] for row in rows}) == [0, 1, 2]
    assert all(row["elapsed_ns"] > 0 for row in rows)
