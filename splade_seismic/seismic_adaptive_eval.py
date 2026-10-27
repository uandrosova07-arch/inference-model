"""SEISMIC без остановки против SEISMIC с остановкой по росту k-го скора (метод search_adaptive из
пропатченной сборки pyseismic-lsr). Обе версии — в ОДНОЙ сборке, на одном индексе и одних запросах.

Остановка: слова запроса обходятся по убыванию веса (не больше query_cut); после каждого слова, начиная
с min_terms-го, если k-й скор в куче вырос относительно меньше чем на stop_eps — обход прекращается.
Параметры (query_cut, heap_factor, stop_eps, min_terms) общие для всех запросов, а останавливается каждый
запрос в свой момент.

Сравнение честное: параметры выбираются на tune (минимум средней задержки при полноте >= цели), числа — на test;
ratio = задержка метода / задержка лучшей конфигурации обычного SEISMIC при той же полноте (фронт Парето на test);
95% bootstrap-интервал для ratio по запросам test.

  python seismic_adaptive_eval.py --docs docs.npz --queries queries.npz --n-postings 1000 --out adapt_1000.npz
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp

CUTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
HFS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
A_CUTS = [5, 8, 10, 15, 20, 30]
A_HFS = [0.8, 0.85, 0.9, 0.95, 1.0]
A_EPS = [0.0, 0.001, 0.003, 0.01, 0.02, 0.05]
A_MIN = [1, 2, 3]


def measure(args, docs, queries):
    import seismic
    assert hasattr(seismic.SeismicIndex, "search_adaptive"), \
        "установлена обычная сборка pyseismic-lsr, нужна пропатченная (с search_adaptive)"
    stype = seismic.get_seismic_string()
    ds = seismic.SeismicDataset(dim=docs.shape[1])
    ds.add_documents_by_id([str(i) for i in range(docs.shape[0])], docs.indices.astype(np.int64),
                           docs.data.astype(np.float32), docs.indptr.astype(np.int64))
    index = seismic.SeismicIndex.build_from_dataset(ds, n_postings=args.n_postings, num_threads=args.build_threads)

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

    # проверка: при выключенной остановке search_adaptive совпадает с search
    for qi in range(min(50, n_q)):
        a = [r[2] for r in index.search(str(qi), comps[qi], vals[qi], args.k, 10, 0.9)]
        b = [r[2] for r in index.search_adaptive(str(qi), comps[qi], vals[qi], args.k, 10, 0.9, float("-inf"), 1)]
        assert a == b, "search_adaptive с выключенной остановкой отличается от search"

    configs = [("base", c, h, np.nan, 0) for c in CUTS for h in HFS]
    configs += [("adapt", c, h, e, m) for c in A_CUTS for h in A_HFS for e in A_EPS for m in A_MIN]
    L = np.zeros((n_q, len(configs)))
    R = np.zeros((n_q, len(configs)))
    for qi in range(n_q):  # прогрев
        index.search(str(qi), comps[qi], vals[qi], args.k, 10, 0.9)
    t_start = time.time()
    # порядок настроек случайный (обычные и адаптивные вперемешку): медленный дрейф нагрузки на общем сервере
    # не должен систематически благоприятствовать одному из методов; контроль дрейфа — повторный замер
    # одной обычной настройки в начале, середине и конце
    order = np.random.default_rng(2026).permutation(len(configs))
    probe_j = next((i for i, c in enumerate(configs) if c[0] == "base" and c[1] == 5 and c[2] == 0.9), 0)
    drift = []

    def probe():
        kind0, c0, h0, _, _ = configs[probe_j]
        t0 = time.perf_counter_ns()
        for qi in range(n_q):
            index.search(str(qi), comps[qi], vals[qi], args.k, c0, h0)
        drift.append((time.perf_counter_ns() - t0) / 1000.0 / n_q)

    probe()
    for step, j in enumerate(order):
        kind, c, h, e, m = configs[j]
        if step == len(order) // 2:
            probe()
        for qi in range(n_q):
            best = np.inf
            for _ in range(args.repeats):
                t0 = time.perf_counter_ns()
                if kind == "base":
                    res = index.search(str(qi), comps[qi], vals[qi], args.k, c, h)
                else:
                    res = index.search_adaptive(str(qi), comps[qi], vals[qi], args.k, c, h, e, m)
                best = min(best, time.perf_counter_ns() - t0)
            L[qi, j] = best / 1000.0
            R[qi, j] = len(exact[qi] & {int(r[2]) for r in res}) / args.k
        if step % 50 == 49:
            print(f"  конфигураций {step + 1}/{len(configs)}, прошло {time.time() - t_start:.0f} с", flush=True)
    probe()
    print("контроль дрейфа (средняя задержка одной обычной настройки в начале/середине/конце, мкс): "
          + " / ".join(f"{x:.1f}" for x in drift)
          + f"; разброс {100 * (max(drift) - min(drift)) / min(drift):.1f}%", flush=True)
    np.savez(args.out, L=L, R=R, kind=np.array([c[0] for c in configs]),
             params=np.array([c[1:] for c in configs], dtype=float), drift=np.array(drift))
    print("замеры сохранены в", args.out, flush=True)


def frontier(rec, lat):
    o = np.argsort(lat)
    fr_r, fr_l, best = [], [], -1.0
    for i in o:
        if rec[i] > best + 1e-12:
            fr_r.append(rec[i]); fr_l.append(lat[i]); best = rec[i]
    return np.array(fr_r), np.array(fr_l)


def interp(fr, target):
    fr_r, fr_l = fr
    if target < fr_r[0] or target > fr_r[-1]:
        return np.nan
    return float(np.interp(target, fr_r, fr_l))


def analyze(args):
    d = np.load(args.out)
    L, R, kind, params = d["L"], d["R"], d["kind"], d["params"]
    n_q = L.shape[0]
    base = np.flatnonzero(kind == "base")
    adapt = np.flatnonzero(kind == "adapt")
    name = lambda j: (f"cut={int(params[j, 0])},hf={params[j, 1]}" if kind[j] == "base" else
                      f"maxcut={int(params[j, 0])},hf={params[j, 1]},eps={params[j, 2]},min={int(params[j, 3])}")
    print(f"queries={n_q} n_postings={args.n_postings}; задержка в микросекундах, один поток\n")

    res = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_q)
        tune, test = np.sort(perm[: n_q // 2]), np.sort(perm[n_q // 2:])
        fr_test = frontier(R[test][:, base].mean(0), L[test][:, base].mean(0))
        for target in args.targets:
            for label, cols in (("base", base), ("adapt", adapt)):
                ok = [j for j in cols if R[tune, j].mean() >= target]
                if not ok:
                    res.setdefault((target, label), []).append(None)
                    continue
                j = min(ok, key=lambda j: L[tune, j].mean())
                r, l = R[test, j].mean(), L[test, j].mean()
                ratio = l / interp(fr_test, r)
                # хвост: p95 выбранной конфигурации против p95 базовой конфигурации, выбранной на tune для той же цели
                jb = min([b for b in base if R[tune, b].mean() >= target], key=lambda b: L[tune, b].mean())
                p95r = np.percentile(L[test, j], 95) / np.percentile(L[test, jb], 95)
                his = []
                for _ in range(args.n_boot):
                    s = test[rng.integers(0, len(test), len(test))]
                    fb = frontier(R[s][:, base].mean(0), L[s][:, base].mean(0))
                    b = interp(fb, R[s, j].mean())
                    if np.isfinite(b) and b > 0:
                        his.append(L[s, j].mean() / b)
                lo, hi = (np.percentile(his, 2.5), np.percentile(his, 97.5)) if len(his) > args.n_boot // 2 else (np.nan, np.nan)
                res.setdefault((target, label), []).append(
                    {"recall": r, "lat": l, "p95": np.percentile(L[test, j], 95), "ratio": ratio,
                     "lo": lo, "hi": hi, "p95r": p95r, "cfg": name(j)})

    print(f"Сводка по seed {args.seeds} (параметры на tune, числа на test).")
    print("ratio — к лучшей конфигурации обычного SEISMIC при той же полноте; [CI] — 95% bootstrap (seed-среднее границ);")
    print("CI<1 — в скольких seed верхняя граница < 1; p95r — p95 задержки к p95 базовой конфигурации для той же цели.\n")
    print(f"{'target':>6} | {'метод':<5} {'recall':>6} {'мкс':>6} {'p95':>6} {'ratio':>6} {'CI':>14} {'CI<1':>5} {'p95r':>5}  config (seed {args.seeds[0]})")
    for target in args.targets:
        for label in ("base", "adapt"):
            v = res[(target, label)]
            if any(x is None for x in v):
                print(f"{target:>6.2f} | {label:<5} недостижимо")
                continue
            g = lambda key: np.array([x[key] for x in v], dtype=float)
            ci = f"[{np.nanmean(g('lo')):.2f}, {np.nanmean(g('hi')):.2f}]"
            print(f"{target:>6.2f} | {label:<5} {g('recall').mean():>6.3f} {g('lat').mean():>6.0f} {g('p95').mean():>6.0f} "
                  f"{np.nanmean(g('ratio')):>6.2f} {ci:>14} {int(np.sum(g('hi') < 1 - 1e-9)):>2}/{len(v):<2} "
                  f"{np.nanmean(g('p95r')):>5.2f}  {v[0]['cfg']}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--n-postings", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.97])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        docs = sp.load_npz(args.docs).tocsr()
        docs.sort_indices()
        queries = sp.load_npz(args.queries).tocsr()
        queries.sort_indices()
        measure(args, docs, queries)
    analyze(args)


if __name__ == "__main__":
    main()
