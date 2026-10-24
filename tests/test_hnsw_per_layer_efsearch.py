# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the per-level efSearch on the HNSW upper levels
(SearchParametersHNSW.efSearch_by_level)."""

import unittest

import faiss
import numpy as np


def make_params(efSearch, efSearch_by_level=None, sel=None):
    params = faiss.SearchParametersHNSW()
    params.efSearch = efSearch
    if efSearch_by_level is not None:
        faiss.copy_array_to_vector(
            np.array(efSearch_by_level, dtype="int32"),
            params.efSearch_by_level,
        )
    if sel is not None:
        params.sel = sel
    return params


def decay_schedule(ef0, r, max_level):
    """ef_l = max(1, floor(ef0 * r^l)) for l >= 1, entry 0 is ef0."""
    return [ef0] + [
        max(1, int(np.floor(ef0 * r ** level)))
        for level in range(1, max_level + 1)
    ]


def search_with_stats(index, xq, k, params):
    stats = faiss.cvar.hnsw_stats
    stats.reset()
    D, I = index.search(xq, k, params=params)
    return D, I, stats.ndis, stats.ndis_upper, stats.nhops


def exhaustive_descent(index, xb, xq):
    """Reference descent where each upper level visits every node reachable
    from all its seeds and hands the whole visited set down, sorted nearest
    first, as seeds of the next level. The upper levels are not always
    strongly connected, so it follows the links instead of brute-forcing
    over the level. Returns, per query, the set handed down to level 0
    (ids and distances) and the number of distances computed on the upper
    levels (the seeds keep their distances)."""
    hnsw = index.hnsw
    offsets = faiss.vector_to_array(hnsw.offsets)
    neighbors = faiss.vector_to_array(hnsw.neighbors)
    cum = faiss.vector_to_array(hnsw.cum_nneighbor_per_level)
    storage = faiss.downcast_index(index.storage)

    def links(i, level):
        nl = neighbors[
            int(offsets[i] + cum[level]):int(offsets[i] + cum[level + 1])
        ]
        return nl[nl >= 0]

    def distances(q, ids):
        dis = np.empty(len(ids), dtype="float32")
        ids = np.ascontiguousarray(ids, dtype="int64")
        storage.compute_distance_subset(
            1, faiss.swig_ptr(q), len(ids),
            faiss.swig_ptr(dis), faiss.swig_ptr(ids),
        )
        return dis

    eps, ep_dis = [], []
    ndis = 0
    for q in xq:
        seeds = [hnsw.entry_point]
        for level in range(hnsw.max_level, 0, -1):
            seen = set(seeds)
            stack = list(seeds)
            while stack:
                u = stack.pop()
                for v in links(u, level):
                    if int(v) not in seen:
                        seen.add(int(v))
                        stack.append(int(v))
            ndis += len(seen) - len(seeds)
            ids = np.array(sorted(seen), dtype="int64")
            dis = distances(q, ids)
            o = np.argsort(dis, kind="stable")
            seeds = [int(i) for i in ids[o]]
        eps.append(np.array(seeds, dtype="int32"))
        ep_dis.append(dis[o])
    return eps, ep_dis, ndis


class TestHNSWPerLayerEfSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Gaussian mixture: clusters create local minima on the upper levels
        d, nb, nq = 16, 5000, 200
        rs = np.random.RandomState(123)
        centers = rs.normal(size=(50, d)) * 4
        spread = rs.uniform(0.3, 1.5, size=50)
        a = rs.randint(50, size=nb + nq)
        x = centers[a] + rs.normal(size=(nb + nq, d)) * spread[a, None]
        x = x.astype("float32")
        cls.xb, cls.xq = x[:nb], x[nb:]
        cls.k = 10

        cls.index = faiss.IndexHNSWFlat(d, 8)
        cls.index.hnsw.efConstruction = 40
        cls.index.add(cls.xb)
        cls.max_level = cls.index.hnsw.max_level
        assert cls.max_level >= 2

        index_flat = faiss.IndexFlatL2(d)
        index_flat.add(cls.xb)
        _, cls.Iref = index_flat.search(cls.xq, cls.k)

    def recall(self, I):
        return np.mean([
            len(set(I[i]) & set(self.Iref[i])) / self.k
            for i in range(len(I))
        ])

    def test_greedy_schedules_match_stock(self):
        # widths <= 1, a short vector and an empty one all keep the
        # stock greedy descent
        index, xq, k = self.index, self.xq, self.k
        for ef0 in 16, 64:
            index.hnsw.efSearch = ef0
            stats = faiss.cvar.hnsw_stats
            stats.reset()
            Dref, Iref = index.search(xq, k)
            ref = (stats.ndis, stats.ndis_upper, stats.nhops)
            self.assertGreater(stats.ndis_upper, 0)
            for schedule in (
                None,
                [ef0] + [1] * self.max_level,
                decay_schedule(ef0, 0.0, self.max_level),
                [ef0, 1],
                [ef0, 0, -1],
            ):
                D, I, *counts = search_with_stats(
                    index, xq, k, make_params(ef0, schedule)
                )
                np.testing.assert_array_equal(I, Iref)
                np.testing.assert_array_equal(D, Dref)
                self.assertEqual(tuple(counts), ref)
        index.hnsw.efSearch = 16

    def test_exhaustive_descent(self):
        # A width >= |S_l| explores everything reachable on each upper
        # level. ndis_upper is then exact, and a visited table that is not
        # reset between levels would give fewer distances.
        index, xb, xq, k = self.index, self.xb, self.xq, self.k
        huge = [64] + [len(xb)] * self.max_level
        _, I, _, ndis_upper, _ = search_with_stats(
            index, xq, k, make_params(64, huge)
        )
        eps, ep_dis, ndis_ref = exhaustive_descent(index, xb, xq)
        self.assertEqual(ndis_upper, ndis_ref)

        # level 0 must be seeded with the whole level-1 set
        params = make_params(64)
        for qi in range(len(xq)):
            q = xq[qi:qi + 1]
            D0 = np.empty((1, k), dtype="float32")
            I0 = np.empty((1, k), dtype="int64")
            index.search_level_0(
                1, faiss.swig_ptr(q), k,
                faiss.swig_ptr(eps[qi]), faiss.swig_ptr(ep_dis[qi]),
                faiss.swig_ptr(D0), faiss.swig_ptr(I0),
                len(eps[qi]), 2, params,
            )
            np.testing.assert_array_equal(I[qi], I0[0])

    def test_recall_and_cost(self):
        index, xq, k = self.index, self.xq, self.k
        for ef0 in 10, 32:
            _, I, ndis_base, upper_base, _ = search_with_stats(
                index, xq, k, make_params(ef0)
            )
            recall_base = self.recall(I)
            for r in 0.2, 0.5:
                schedule = decay_schedule(ef0, r, self.max_level)
                _, I, ndis, upper, _ = search_with_stats(
                    index, xq, k, make_params(ef0, schedule)
                )
                self.assertGreaterEqual(self.recall(I), recall_base - 0.02)
                if schedule[1] > 1:
                    # wider upper levels cost more there, counted in ndis
                    self.assertGreater(upper, upper_base)
                    self.assertGreaterEqual(
                        ndis - upper, 0.5 * (ndis_base - upper_base)
                    )
        _, I, _, _, _ = search_with_stats(
            index, xq, k,
            make_params(32, decay_schedule(32, 0.5, self.max_level)),
        )
        self.assertGreater(self.recall(I), 0.9)

    def test_no_dis_check(self):
        index, xq, k = self.index, self.xq, self.k
        schedule = decay_schedule(32, 0.5, self.max_level)
        params = make_params(32, schedule)
        params.check_relative_distance = False
        _, I = index.search(xq, k, params=params)
        self.assertGreater(self.recall(I), 0.9)

    def test_unbounded_queue(self):
        index, xq, k = self.index, self.xq, self.k
        params = make_params(32, decay_schedule(32, 0.5, self.max_level))
        params.bounded_queue = False
        _, I = index.search(xq, k, params=params)
        self.assertGreater(self.recall(I), 0.9)

    def test_threads(self):
        index, xq, k = self.index, self.xq, self.k
        params = make_params(32, decay_schedule(32, 0.5, self.max_level))
        nt = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            D1, I1 = index.search(xq, k, params=params)
        finally:
            faiss.omp_set_num_threads(nt)
        D, I = index.search(xq, k, params=params)
        np.testing.assert_array_equal(I, I1)
        np.testing.assert_array_equal(D, D1)

    def test_selector(self):
        # the IDSelector filters the results, not the upper-level descent
        index, xb, xq, k = self.index, self.xb, self.xq, self.k
        even = np.arange(0, len(xb), 2).astype("int64")
        sel = faiss.IDSelectorBatch(even)
        schedule = decay_schedule(32, 0.5, self.max_level)
        _, I = index.search(xq, k, params=make_params(32, schedule, sel))
        self.assertTrue(np.all(I % 2 == 0))
        index_flat = faiss.IndexFlatL2(xb.shape[1])
        index_flat.add(xb[even])
        _, Iref = index_flat.search(xq, k)
        Iref = even[Iref]
        recall = np.mean([
            len(set(I[i]) & set(Iref[i])) / k for i in range(len(I))
        ])
        self.assertGreater(recall, 0.8)


class TestHNSWPerLayerEfSearchIP(unittest.TestCase):

    def test_ip(self):
        # similarity metrics take the other comparator in the same code
        d, nb, nq, k = 16, 3000, 100, 10
        rs = np.random.RandomState(456)
        xb = rs.normal(size=(nb, d)).astype("float32")
        xq = rs.normal(size=(nq, d)).astype("float32")
        index = faiss.IndexHNSWFlat(d, 8, faiss.METRIC_INNER_PRODUCT)
        index.add(xb)
        max_level = index.hnsw.max_level

        Dref, Iref = index.search(xq, k, params=make_params(32))
        D, I = index.search(
            xq, k, params=make_params(32, [32] + [1] * max_level)
        )
        np.testing.assert_array_equal(I, Iref)
        np.testing.assert_array_equal(D, Dref)

        index_flat = faiss.IndexFlatIP(d)
        index_flat.add(xb)
        _, Igt = index_flat.search(xq, k)
        _, I = index.search(
            xq, k, params=make_params(32, decay_schedule(32, 0.5, max_level))
        )
        recall = np.mean([
            len(set(I[i]) & set(Igt[i])) / k for i in range(nq)
        ])
        self.assertGreater(recall, 0.8)


def record_per_query_stats(index, xq, k, params):
    """search with hnsw_per_query_stats enabled. Returns D, I and the
    recorded columns, reshaped per query."""
    s = faiss.cvar.hnsw_per_query_stats
    s.enabled = True
    try:
        s.reset()
        D, I = index.search(xq, k, params=params)
    finally:
        s.enabled = False
    rec = dict(
        level_offsets=faiss.vector_to_array(s.level_offsets),
        eps=faiss.vector_to_array(s.level_eps).reshape(s.nq, s.S),
        ep_dis=faiss.vector_to_array(s.level_ep_dis).reshape(s.nq, s.S),
        ndis=faiss.vector_to_array(s.ndis).reshape(s.nq, s.L),
        nhops=faiss.vector_to_array(s.nhops).reshape(s.nq, s.L),
    )
    return D, I, rec


class TestHNSWPerQueryStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d, nb, nq = 16, 5000, 200
        rs = np.random.RandomState(123)
        centers = rs.normal(size=(50, d)) * 4
        spread = rs.uniform(0.3, 1.5, size=50)
        a = rs.randint(50, size=nb + nq)
        x = centers[a] + rs.normal(size=(nb + nq, d)) * spread[a, None]
        x = x.astype("float32")
        cls.xb, cls.xq = x[:nb], x[nb:]
        cls.k = 10
        cls.index = faiss.IndexHNSWFlat(d, 8)
        cls.index.add(cls.xb)
        cls.max_level = cls.index.hnsw.max_level
        assert cls.max_level >= 2
        cls.levels = faiss.vector_to_array(cls.index.hnsw.levels)
        cls.schedules = {
            "greedy": None,
            "decay": decay_schedule(32, 0.5, cls.max_level),
            "beam": [32] + [8] * cls.max_level,
            # a greedy level below a beam level, the levels past the end
            # of the vector are greedy as well
            "mixed": [32, 1, 6],
        }

    def tearDown(self):
        s = faiss.cvar.hnsw_per_query_stats
        s.enabled = False
        s.reset()

    def caps(self, schedule):
        sched = schedule or []
        return [0] + [
            max(1, sched[level]) if level < len(sched) else 1
            for level in range(1, self.max_level + 1)
        ]

    def test_results_unchanged(self):
        index, xq, k = self.index, self.xq, self.k
        hs = faiss.cvar.hnsw_stats
        for schedule in self.schedules.values():
            params = make_params(32, schedule)
            hs.reset()
            Dref, Iref = index.search(xq, k, params=params)
            ref = (hs.n1, hs.n2, hs.ndis, hs.ndis_upper, hs.nhops)
            hs.reset()
            D, I, _ = record_per_query_stats(index, xq, k, params)
            np.testing.assert_array_equal(I, Iref)
            np.testing.assert_array_equal(D, Dref)
            self.assertEqual(
                (hs.n1, hs.n2, hs.ndis, hs.ndis_upper, hs.nhops), ref
            )

    def test_counts_sum_to_hnsw_stats(self):
        index, xq, k = self.index, self.xq, self.k
        hs = faiss.cvar.hnsw_stats
        for schedule in self.schedules.values():
            hs.reset()
            _, _, rec = record_per_query_stats(
                index, xq, k, make_params(32, schedule)
            )
            ndis, nhops = rec["ndis"], rec["nhops"]
            self.assertEqual(ndis.shape, (len(xq), self.max_level + 1))
            self.assertEqual(ndis.sum(), hs.ndis)
            self.assertEqual(ndis[:, 1:].sum(), hs.ndis_upper)
            self.assertEqual(nhops.sum(), hs.nhops)
            self.assertTrue(np.all(ndis[:, 0] > 0))
            self.assertTrue(np.all(nhops >= 1))

    def test_handed_down_sets(self):
        index, xq, k = self.index, self.xq, self.k
        storage = faiss.downcast_index(index.storage)
        for name, schedule in self.schedules.items():
            _, _, rec = record_per_query_stats(
                index, xq, k, make_params(32, schedule)
            )
            caps = self.caps(schedule)
            lo = rec["level_offsets"]
            np.testing.assert_array_equal(lo, np.cumsum([0] + caps))
            eps, ep_dis = rec["eps"], rec["ep_dis"]
            self.assertEqual(eps.shape, (len(xq), lo[-1]))
            for qi in range(len(xq)):
                prev_best = None
                for level in range(self.max_level, 0, -1):
                    ids = eps[qi, lo[level]:lo[level + 1]]
                    dis = ep_dis[qi, lo[level]:lo[level + 1]]
                    n = int(np.sum(ids >= 0))
                    self.assertGreater(n, 0)
                    # padding only at the end
                    self.assertTrue(np.all(ids[:n] >= 0))
                    self.assertTrue(np.all(ids[n:] == -1))
                    self.assertTrue(np.all(np.isinf(dis[n:])))
                    if caps[level] == 1:
                        self.assertEqual(len(ids), 1)
                    ids, dis = ids[:n], dis[:n]
                    self.assertEqual(len(set(ids.tolist())), n)
                    self.assertTrue(np.all(np.diff(dis) >= 0), name)
                    # the nodes live on that level
                    self.assertTrue(np.all(self.levels[ids] > level))
                    # distances as recomputed from the storage
                    ref = np.empty(n, dtype="float32")
                    ids64 = ids.astype("int64")
                    storage.compute_distance_subset(
                        1, faiss.swig_ptr(xq[qi]), n,
                        faiss.swig_ptr(ref), faiss.swig_ptr(ids64),
                    )
                    np.testing.assert_allclose(dis, ref, rtol=1e-5)
                    # the best of the set never gets worse going down
                    if prev_best is not None:
                        self.assertLessEqual(dis[0], prev_best)
                    prev_best = dis[0]

    def test_level_0_replay(self):
        # level 0 seeded with the recorded level-1 set gives the results
        index, xq, k = self.index, self.xq, self.k
        for name in "greedy", "decay", "beam":
            params = make_params(32, self.schedules[name])
            _, I, rec = record_per_query_stats(index, xq, k, params)
            lo = rec["level_offsets"]
            params0 = make_params(32)
            for qi in range(len(xq)):
                ids = rec["eps"][qi, lo[1]:lo[2]]
                dis = rec["ep_dis"][qi, lo[1]:lo[2]]
                n = int(np.sum(ids >= 0))
                ids = np.ascontiguousarray(ids[:n])
                dis = np.ascontiguousarray(dis[:n])
                D0 = np.empty((1, k), dtype="float32")
                I0 = np.empty((1, k), dtype="int64")
                index.search_level_0(
                    1, faiss.swig_ptr(xq[qi:qi + 1]), k,
                    faiss.swig_ptr(ids), faiss.swig_ptr(dis),
                    faiss.swig_ptr(D0), faiss.swig_ptr(I0),
                    n, 2, params0,
                )
                np.testing.assert_array_equal(I[qi], I0[0], name)

    def test_threads(self):
        index, xq, k = self.index, self.xq, self.k
        params = make_params(32, self.schedules["decay"])
        nt = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            _, _, rec1 = record_per_query_stats(index, xq, k, params)
        finally:
            faiss.omp_set_num_threads(nt)
        _, _, rec = record_per_query_stats(index, xq, k, params)
        for key in rec:
            np.testing.assert_array_equal(rec[key], rec1[key], key)

    def test_append_reset_and_layout(self):
        index, xq, k = self.index, self.xq, self.k
        params = make_params(32, self.schedules["decay"])
        _, _, rec_a = record_per_query_stats(index, xq[:50], k, params)
        _, _, rec_b = record_per_query_stats(index, xq[50:120], k, params)
        s = faiss.cvar.hnsw_per_query_stats
        s.enabled = True
        try:
            s.reset()
            index.search(xq[:50], k, params=params)
            index.search(xq[50:120], k, params=params)
            self.assertEqual(s.nq, 120)
            eps = faiss.vector_to_array(s.level_eps).reshape(s.nq, s.S)
            ndis = faiss.vector_to_array(s.ndis).reshape(s.nq, s.L)
            np.testing.assert_array_equal(
                eps, np.vstack([rec_a["eps"], rec_b["eps"]])
            )
            np.testing.assert_array_equal(
                ndis, np.vstack([rec_a["ndis"], rec_b["ndis"]])
            )

            # a different schedule changes the layout
            S = s.S
            with self.assertRaises(RuntimeError):
                index.search(
                    xq[:5], k,
                    params=make_params(32, self.schedules["beam"]),
                )
            self.assertEqual((s.nq, s.S), (120, S))

            # so does a different max_level
            other = faiss.IndexHNSWFlat(self.xb.shape[1], 32)
            other.add(self.xb[:200])
            self.assertNotEqual(other.hnsw.max_level, self.max_level)
            with self.assertRaises(RuntimeError):
                other.search(xq[:5], k, params=make_params(32))

            s.reset()
            self.assertEqual((s.nq, s.L, s.S), (0, 0, 0))
            self.assertEqual(s.level_eps.size(), 0)
            self.assertEqual(s.ndis.size(), 0)
            self.assertEqual(s.level_offsets.size(), 0)
            other.search(xq[:5], k, params=make_params(32))
            self.assertEqual((s.nq, s.L), (5, other.hnsw.max_level + 1))
        finally:
            s.enabled = False
            s.reset()
