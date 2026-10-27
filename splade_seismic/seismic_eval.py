"""Настоящий SEISMIC (pip-пакет pyseismic-lsr): реальная задержка и полнота для каждого запроса.

Что делает:
  1) строит (или загружает) индекс SEISMIC по docs.npz;
  2) для сетки query_cut x heap_factor выполняет КАЖДЫЙ запрос по отдельности и замеряет время
     (минимум из --repeats повторов, один поток) и полноту относительно точного top-k;
  3) сравнивает: одна общая конфигурация для всех запросов против выбора конфигурации для каждого запроса
     (правило по доле массы, линейная модель, оракул). Параметры выбираются на tune, числа — на test.
Сырые матрицы сохраняются в --out (npz), чтобы анализ можно было повторить без замеров: --analyze-only.

Пакет собран только для Python 3.10/3.11 (на момент написания), см. инструкцию к запуску.

  python seismic_eval.py --docs docs.npz --queries queries.npz --n-postings 1000 --index idx_1000 --out seismic_1000.npz
"""
import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_cut_v2 import coord_stats, query_features

CUTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
HFS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]


def build_or_load(docs, n_postings, index_path, threads):
    import seismic
    if index_path and os.path.exists(index_path):
        print("загружаю индекс", index_path, flush=True)
        return seismic.SeismicIndex.load(index_path)
    ds = seismic.SeismicDataset(dim=docs.shape[1])
    ds.add_documents_by_id([str(i) for i in range(docs.shape[0])], docs.indices.astype(np.int64),
                           docs.data.astype(np.float32), docs.indptr.astype(np.int64))
    t0 = time.time()
    index = seismic.SeismicIndex.build_from_dataset(ds, n_postings=n_postings, num_threads=threads)
    print(f"индекс построен за {time.time() - t0:.0f} с", flush=True)
    if index_path:
        index.save(index_path)
    return index


def measure(args, docs, queries):
    import seismic
    stype = seismic.get_seismic_string()
    index = build_or_load(docs, args.n_postings, args.index, args.build_threads)
    docs_csc = docs.tocsc()
    n_q = queries.shape[0]
    comps, vals, exact = [], [], []
    for qi in range(n_q):
        s, e = queries.indptr[qi], queries.indptr[qi + 1]
        idx, val = queries.indices[s:e], queries.data[s:e].astype(np.float32)
        comps.append(np.array([str(i) for i in idx], dtype=stype))
        vals.append(val)
        scores = np.asarray(docs_csc[:, idx] @ val.astype(np.float64)).ravel()
        exact.append(set(np.argpartition(-scores, args.k)[:args.k].tolist()))

    configs = [(c, h) for c in CUTS for h in HFS]
    L = np.zeros((n_q, len(configs)))
    R = np.zeros((n_q, len(configs)))
    for qi in range(n_q):  # прогрев
        index.search(str(qi), comps[qi], vals[qi], args.k, 10, 0.9)
    t_start = time.time()
    for j, (c, h) in enumerate(configs):
        for qi in range(n_q):
            best = np.inf
            for _ in range(args.repeats):
                t0 = time.perf_counter_ns()
                res = index.search(str(qi), comps[qi], vals[qi], args.k, c, h)
                best = min(best, time.perf_counter_ns() - t0)
            L[qi, j] = best / 1000.0  # микросекунды
            R[qi, j] = len(exact[qi] & {int(r[2]) for r in res}) / args.k
        if j % 8 == 7:
            print(f"  конфигураций {j + 1}/{len(configs)}, прошло {time.time() - t_start:.0f} с", flush=True)

    # ориентир: точный перебор в том же движке (на части запросов)
    ds = seismic.SeismicDataset(dim=docs.shape[1])
    ds.add_documents_by_id([str(i) for i in range(docs.shape[0])], docs.indices.astype(np.int64),
                           docs.data.astype(np.float32), docs.indptr.astype(np.int64))
    bf = []
    for qi in range(min(50, n_q)):
        t0 = time.perf_counter_ns()
        ds.search(str(qi), comps[qi], vals[qi], args.k)
        bf.append((time.perf_counter_ns() - t0) / 1000.0)
    np.savez(args.out, L=L, R=R, configs=np.array(configs), brute_us=np.array(bf))
    print("замеры сохранены в", args.out, flush=True)


def frontier_interp(rec, lat, target):
    """Задержка лучшей ОБЩЕЙ конфигурации при средней полноте target (линейная интерполяция по фронту Парето)."""
    o = np.argsort(lat)
    fr_r, fr_l, best = [], [], -1.0
    for i in o:
        if rec[i] > best + 1e-12:
            fr_r.append(rec[i]); fr_l.append(lat[i]); best = rec[i]
    fr_r, fr_l = np.array(fr_r), np.array(fr_l)
    if target < fr_r[0] or target > fr_r[-1]:
        return np.nan
    return float(np.interp(target, fr_r, fr_l))


def analyze(args, docs, queries):
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)  # nan там, где полнота вне диапазона общих конфигураций
    d = np.load(args.out)
    L, R, configs = d["L"], d["R"], [tuple(c) for c in d["configs"]]
    n_q, n_c = L.shape
    cfg_index = {(int(c), float(h)): j for j, (c, h) in enumerate(configs)}
    k = args.k
    print(f"docs={docs.shape[0]} queries={n_q} n_postings={args.n_postings}; задержка в микросекундах, один поток")
    print(f"точный перебор в том же движке: медиана {np.median(d['brute_us']):.0f} мкс на запрос")

    df, maxw, thr = coord_stats(docs, args.n_postings)
    feats, qmass = [], []
    for qi in range(n_q):
        s, e = queries.indptr[qi], queries.indptr[qi + 1]
        idx, val = queries.indices[s:e], queries.data[s:e].astype(np.float64)
        o = np.argsort(-val)
        idx, val = idx[o], val[o]
        feats.append(query_features(idx, val, df, maxw, thr, docs.shape[0]))
        qmass.append(np.cumsum(val) / val.sum())
    feats = np.array(feats)

    def snap_cut(m):
        """Наименьший cut из сетки, не меньший m."""
        i = np.searchsorted(CUTS, m)
        return CUTS[min(i, len(CUTS) - 1)]

    methods = ["fixed", "gamma", "linear", "oracle"]
    summary = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_q)
        tune, test = np.sort(perm[: n_q // 2]), np.sort(perm[n_q // 2:])
        cand = {m: [] for m in methods}  # (params, j_per_query)
        for j, (c, h) in enumerate(configs):
            cand["fixed"].append((f"cut={int(c)},hf={h}", np.full(n_q, j)))
        for h in HFS:
            for g in np.linspace(0.05, 1.0, 39):
                js = [cfg_index[(snap_cut(int(np.searchsorted(m, g)) + 1), h)] for m in qmass]
                cand["gamma"].append((f"gamma={g:.2f},hf={h}", np.array(js)))
            for tau in (0.8, 0.9, 1.0):
                cols = [cfg_index[(c, h)] for c in CUTS]
                ok = R[:, cols] >= tau - 1e-12
                y = np.log(np.where(ok.any(1), np.array(CUTS)[ok.argmax(1)], CUTS[-1]))
                pred = np.zeros(n_q)
                mk = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                for tr, va in KFold(5, shuffle=True, random_state=seed).split(tune):
                    pred[tune[va]] = mk().fit(feats[tune[tr]], y[tune[tr]]).predict(feats[tune[va]])
                pred[test] = mk().fit(feats[tune], y[tune]).predict(feats[test])
                for dlt in np.linspace(-1.0, 2.0, 16):
                    js = [cfg_index[(snap_cut(np.exp(p + dlt)), h)] for p in pred]
                    cand["linear"].append((f"hf={h},tau={tau},d={dlt:+.1f}", np.array(js)))
        for tau in np.linspace(0.5, 1.0, 11):
            ok = R >= tau - 1e-12
            lat_ok = np.where(ok, L, np.inf)
            js = np.where(ok.any(1), lat_ok.argmin(1), R.argmax(1))
            cand["oracle"].append((f"recall_q>={tau:.2f}", js))

        rows = np.arange(n_q)
        mean_r_test, mean_l_test = R[test].mean(0), L[test].mean(0)
        p95_l_test = np.percentile(L[test], 95, axis=0)
        for target in args.targets:
            for m in methods:
                part = test if m == "oracle" else tune
                scored = []
                for params, js in cand[m]:
                    r, l = R[rows, js], L[rows, js]
                    if r[part].mean() >= target:
                        scored.append((l[part].mean(), params, r, l))
                if not scored:
                    summary.setdefault((target, m), []).append(None)
                    continue
                _, params, r, l = min(scored, key=lambda x: x[0])
                rt, lt = r[test].mean(), l[test].mean()
                base = frontier_interp(mean_r_test, mean_l_test, rt)
                base95 = frontier_interp(mean_r_test, p95_l_test, rt)
                summary.setdefault((target, m), []).append({
                    "recall": rt, "lat": lt, "p95": np.percentile(l[test], 95), "params": params,
                    "ratio": lt / base if base else np.nan,
                    "p95r": np.percentile(l[test], 95) / base95 if base95 else np.nan})

    print(f"\nСводка по seed {args.seeds}: параметры на tune, числа на test.")
    print("ratio = средняя задержка / задержка лучшей общей конфигурации при той же полноте (фронт Парето на test);")
    print("p95r — то же для 95-го перцентиля задержки. config — выбранные параметры в первом seed.\n")
    print(f"{'target':>6} | {'метод':<7} {'recall':>6} {'мкс':>7} {'p95':>7} {'ratio':>6} {'p95r':>6}  config(seed {args.seeds[0]})")
    for target in args.targets:
        for m in methods:
            vals = summary[(target, m)]
            if any(v is None for v in vals):
                print(f"{target:>6.2f} | {m:<7} недостижимо хотя бы в одном seed")
                continue
            g = lambda key: np.array([v[key] for v in vals], dtype=float)
            print(f"{target:>6.2f} | {m:<7} {g('recall').mean():>6.3f} {g('lat').mean():>7.0f} {g('p95').mean():>7.0f} "
                  f"{np.nanmean(g('ratio')):>6.2f} {np.nanmean(g('p95r')):>6.2f}  {vals[0]['params']}")
        print()
    print("fixed может давать ratio > 1: выбор на tune не всегда попадает на фронт Парето test. Оракул нереализуем;")
    print("nan — полнота метода выше, чем у любой общей конфигурации, сравнивать не с чем.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--n-postings", type=int, default=1000)
    ap.add_argument("--index", help="путь для сохранения/загрузки индекса")
    ap.add_argument("--out", required=True, help="npz с замерами")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.97])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--analyze-only", action="store_true", help="не мерить, только проанализировать --out")
    args = ap.parse_args()

    docs = sp.load_npz(args.docs).tocsr()
    docs.sort_indices()
    queries = sp.load_npz(args.queries).tocsr()
    if args.max_queries:
        queries = queries[: args.max_queries]
    queries.sort_indices()
    if not args.analyze_only:
        measure(args, docs, queries)
    analyze(args, docs, queries)


if __name__ == "__main__":
    main()
