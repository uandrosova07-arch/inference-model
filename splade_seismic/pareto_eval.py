"""Адаптивный SEISMIC против «усиления» гиперпараметров обычного SEISMIC: что теряет меньше качества за то же ускорение.

Измерение (env311, нужна пропатченная сборка): один индекс, одни запросы; для каждой настройки обычного SEISMIC
(query_cut x heap_factor) и адаптивного (max query_cut x heap_factor x stop_eps x min_terms) — задержка каждого
запроса (минимум из --repeats) и его top-10. Сохраняет npz.
Анализ (любое окружение с numpy; график — если есть matplotlib):
  1. Ускорение -> потеря качества. Опора — самая точная настройка обычного SEISMIC (максимальная полнота).
     Для каждого метода — фронт Парето «ускорение относительно опоры — качество» (полнота@10 к точному поиску и
     MRR@10 по разметке dev) на всех запросах; таблица потерь качества при ускорении x1.5, x2, x3, x4, x6.
     Отдельно — «одиночные ручки»: обычный SEISMIC при изменении только query_cut или только heap_factor
     и адаптивный при изменении только stop_eps.
  2. Честная проверка (tune/test, 3 seed): для бюджетов задержки параметры каждого метода выбираются на tune
     (наилучшее качество при задержке <= бюджета), качество и задержка — на test; разность качества
     adapt - base с парным bootstrap-интервалом по запросам test.
  3. Обратная сторона: задержка при равной полноте (как в seismic_adaptive_eval.py) — для полноты.

  python pareto_eval.py --docs ENC/docs.npz --queries ENC/queries.npz --sub SUB --zip msmarco.zip --out pareto_if.npz
  python pareto_eval.py ... --out pareto_if.npz --analyze-only --name inference-free
"""
import argparse
import json
import os
import time
import zipfile

import numpy as np
import scipy.sparse as sp

CUTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
HFS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
A_CUTS = [5, 8, 10, 15, 20, 30]
A_HFS = [0.8, 0.85, 0.9, 0.95, 1.0]
A_EPS = [0.0, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1]
A_MIN = [1, 2, 3]
SPEEDUPS = [1.5, 2, 3, 4, 6]


def read_jsonl(path, key):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[key] for line in f]


def load_rel(sub, zip_path):
    q_ids = read_jsonl(os.path.join(sub, "queries.jsonl"), "_id")
    doc_ids = read_jsonl(os.path.join(sub, "corpus.jsonl"), "_id")
    pos = {d: i for i, d in enumerate(doc_ids)}
    rel = [set() for _ in q_ids]
    qi = {q: i for i, q in enumerate(q_ids)}
    with zipfile.ZipFile(zip_path) as z:
        with z.open(next(n for n in z.namelist() if n.endswith("qrels/dev.tsv"))) as f:
            next(f)
            for line in f:
                q, d, s = line.decode("utf-8").rstrip("\n").split("\t")
                if int(s) > 0 and q in qi and d in pos:
                    rel[qi[q]].add(pos[d])
    return rel


def measure(args, docs, queries):
    import seismic
    assert hasattr(seismic.SeismicIndex, "search_adaptive"), "нужна пропатченная сборка SEISMIC (search_adaptive)"
    stype = seismic.get_seismic_string()
    ds = seismic.SeismicDataset(dim=docs.shape[1])
    ds.add_documents_by_id([str(i) for i in range(docs.shape[0])], docs.indices.astype(np.int64),
                           docs.data.astype(np.float32), docs.indptr.astype(np.int64))
    index = seismic.SeismicIndex.build_from_dataset(ds, n_postings=args.n_postings, num_threads=args.build_threads)
    docs_csc = docs.tocsc()
    n_q, k = queries.shape[0], args.k
    comps, vals, exact = [], [], np.zeros((n_q, k), dtype=np.int32)
    for qi in range(n_q):
        s, e = queries.indptr[qi], queries.indptr[qi + 1]
        idx, val = queries.indices[s:e], queries.data[s:e].astype(np.float32)
        comps.append(np.array([str(i) for i in idx], dtype=stype))
        vals.append(val)
        sc = np.asarray(docs_csc[:, idx] @ val.astype(np.float64)).ravel()
        t = np.argpartition(-sc, k)[:k]
        exact[qi] = t[np.argsort(-sc[t])]
    for qi in range(min(50, n_q)):
        a = [r[2] for r in index.search(str(qi), comps[qi], vals[qi], k, 10, 0.9)]
        b = [r[2] for r in index.search_adaptive(str(qi), comps[qi], vals[qi], k, 10, 0.9, float("-inf"), 1)]
        assert a == b, "search_adaptive с выключенной остановкой отличается от search"

    configs = [("base", c, h, np.nan, 0) for c in CUTS for h in HFS]
    configs += [("adapt", c, h, e, m) for c in A_CUTS for h in A_HFS for e in A_EPS for m in A_MIN]
    L = np.zeros((n_q, len(configs)), dtype=np.float32)
    top = np.full((len(configs), n_q, k), -1, dtype=np.int32)
    for qi in range(n_q):
        index.search(str(qi), comps[qi], vals[qi], k, 10, 0.9)
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
                    res = index.search(str(qi), comps[qi], vals[qi], k, c, h)
                else:
                    res = index.search_adaptive(str(qi), comps[qi], vals[qi], k, c, h, e, m)
                best = min(best, time.perf_counter_ns() - t0)
            L[qi, j] = best / 1000.0
            res = sorted(res, key=lambda r: -r[1])[:k]
            top[j, qi, :len(res)] = [int(r[2]) for r in res]
        if step % 50 == 49:
            print(f"  конфигураций {step + 1}/{len(configs)}, прошло {time.time() - t_start:.0f} с", flush=True)
    probe()
    print("контроль дрейфа (средняя задержка одной обычной настройки в начале/середине/конце, мкс): "
          + " / ".join(f"{x:.1f}" for x in drift)
          + f"; разброс {100 * (max(drift) - min(drift)) / min(drift):.1f}%", flush=True)
    np.savez_compressed(args.out, L=L, top=top, exact=exact, kind=np.array([c[0] for c in configs]),
                        params=np.array([c[1:] for c in configs], dtype=float), drift=np.array(drift))
    print("замеры сохранены в", args.out, flush=True)


def per_query_quality(top, exact, rel, k):
    """top: C x Q x k -> полнота к точному top-k и reciprocal rank (только запросы с разметкой), C x Q."""
    C, Q, _ = top.shape
    rec = np.zeros((C, Q), dtype=np.float32)
    rr = np.zeros((C, Q), dtype=np.float32)
    ex = [set(e.tolist()) for e in exact]
    for j in range(C):
        for q in range(Q):
            t = top[j, q]
            rec[j, q] = len(ex[q] & set(t.tolist())) / k
            for r, d in enumerate(t):
                if d in rel[q]:
                    rr[j, q] = 1.0 / (r + 1)
                    break
    return rec, rr


def front_by_latency(lat, qual):
    """Ступенчатый фронт: для каждой задержки — лучшее качество среди настроек не медленнее. Возвращает (lat, qual)."""
    o = np.argsort(lat)
    fl, fq, best = [], [], -np.inf
    for i in o:
        if qual[i] > best + 1e-12:
            fl.append(lat[i]); fq.append(qual[i]); best = qual[i]
    return np.array(fl), np.array(fq)


def best_under(fl, fq, budget):
    i = np.searchsorted(fl, budget, side="right") - 1
    return fq[i] if i >= 0 else np.nan


def analyze(args):
    d = np.load(args.out)
    L, top, exact, kind, params = d["L"], d["top"], d["exact"], d["kind"], d["params"]
    rel = load_rel(args.sub, args.zip)
    has_rel = np.array([len(r) > 0 for r in rel])
    rec, rr = per_query_quality(top, exact, rel, args.k)
    base, adapt = np.flatnonzero(kind == "base"), np.flatnonzero(kind == "adapt")
    name = lambda j: (f"cut={int(params[j, 0])},hf={params[j, 1]:g}" if kind[j] == "base" else
                      f"maxcut={int(params[j, 0])},hf={params[j, 1]:g},eps={params[j, 2]:g},min={int(params[j, 3])}")
    lat_m = L.mean(0)
    rec_m = rec.mean(1)
    mrr_m = rr[:, has_rel].mean(1)
    ref = base[np.argmax(rec_m[base] - 1e-9 * lat_m[base])]
    exact_mrr = None
    ex_rr = np.zeros(len(rel))
    for q, t in enumerate(exact):
        for r, dd in enumerate(t):
            if dd in rel[q]:
                ex_rr[q] = 1.0 / (r + 1)
                break
    exact_mrr = ex_rr[has_rel].mean()

    out = []
    pr = lambda s="": (print(s), out.append(s))
    pr(f"=== {args.name}: адаптивный SEISMIC против подбора гиперпараметров обычного SEISMIC ===")
    pr(f"запросов {L.shape[0]} (с разметкой {has_rel.sum()}); настроек: обычный {len(base)}, адаптивный {len(adapt)}")
    pr(f"опора (самая точная обычная настройка): {name(ref)} — {lat_m[ref]:.0f} мкс, полнота {rec_m[ref]:.4f}, "
       f"MRR@10 {mrr_m[ref]:.4f}; точный поиск: MRR@10 {exact_mrr:.4f}")

    # 1. ускорение -> потеря качества, на всех запросах
    pr("\n1. Потеря качества при заданном ускорении относительно опоры (лучшая настройка каждого метода, все запросы)")
    pr(f"{'ускорение':>9} | {'полнота: обычный':>16} {'адаптивный':>10} {'разница':>8} | "
       f"{'MRR@10: обычный':>15} {'адаптивный':>10} {'разница':>8}")
    fronts = {}
    for lab, cols in (("base", base), ("adapt", adapt)):
        fronts[lab, "rec"] = front_by_latency(lat_m[cols], rec_m[cols])
        fronts[lab, "mrr"] = front_by_latency(lat_m[cols], mrr_m[cols])
    for s in SPEEDUPS:
        b = lat_m[ref] / s
        rb, ra = best_under(*fronts["base", "rec"], b), best_under(*fronts["adapt", "rec"], b)
        mb, ma = best_under(*fronts["base", "mrr"], b), best_under(*fronts["adapt", "mrr"], b)
        pr(f"{'x' + format(s, 'g'):>9} | {rb:>16.4f} {ra:>10.4f} {ra - rb:>+8.4f} | {mb:>15.4f} {ma:>10.4f} {ma - mb:>+8.4f}")

    pr("\n   Одиночные ручки (как падает полнота, если «усиливать» один параметр):")
    knobs = [("обычный: только query_cut (hf=1.0)", [j for j in base if params[j, 1] == 1.0], 0),
             ("обычный: только heap_factor (cut=10)", [j for j in base if params[j, 0] == 10], 1)]
    ab = max(adapt, key=lambda j: (rec_m[j] >= rec_m[ref] - 0.005) * 1e6 - lat_m[j])
    knobs.append((f"адаптивный: только stop_eps ({name(ab).split(',eps')[0]},min={int(params[ab, 3])})",
                  [j for j in adapt if (params[j, 0], params[j, 1], params[j, 3]) == (params[ab, 0], params[ab, 1], params[ab, 3])], 2))
    for title, cols, pi in knobs:
        cols = sorted(cols, key=lambda j: -lat_m[j])
        pr(f"   {title}")
        pr("     " + "  ".join(f"{params[j, pi]:g}: x{lat_m[ref] / lat_m[j]:.2f}/{rec_m[j]:.3f}" for j in cols))
    pr("     (значение параметра: ускорение относительно опоры / полнота)")

    # 2. честная проверка
    pr("\n2. Честная проверка: параметры по бюджету задержки выбраны на tune, числа на test (среднее по seed);")
    pr("   разность качества adapt - base, 95% парный bootstrap по запросам test (среднее границ по seed)")
    fb_all = fronts["base", "rec"][0]
    budgets = np.unique(np.round(np.geomspace(max(fb_all.min() * 1.05, 1), lat_m[ref], 8)))
    rows = {}
    rng0 = np.random.default_rng(12345)
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(L.shape[0])
        tune, test = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
        test_rel = test[has_rel[test]]
        for B in budgets:
            pick = {}
            for lab, cols in (("base", base), ("adapt", adapt)):
                ok = [j for j in cols if L[tune, j].mean() <= B]
                pick[lab] = max(ok, key=lambda j: rec[j, tune].mean()) if ok else None
            if pick["base"] is None or pick["adapt"] is None:
                continue
            jb, ja = pick["base"], pick["adapt"]
            dr = rec[ja, test] - rec[jb, test]
            dm = rr[ja, test_rel] - rr[jb, test_rel]
            boot_r = [dr[rng0.integers(0, len(dr), len(dr))].mean() for _ in range(args.n_boot)]
            boot_m = [dm[rng0.integers(0, len(dm), len(dm))].mean() for _ in range(args.n_boot)]
            rows.setdefault(B, []).append({
                "lat_b": L[test, jb].mean(), "lat_a": L[test, ja].mean(),
                "rec_b": rec[jb, test].mean(), "rec_a": rec[ja, test].mean(),
                "mrr_b": rr[jb, test_rel].mean(), "mrr_a": rr[ja, test_rel].mean(),
                "dr": dr.mean(), "dr_ci": np.percentile(boot_r, [2.5, 97.5]),
                "dm": dm.mean(), "dm_ci": np.percentile(boot_m, [2.5, 97.5]),
                "sig_r": int(np.percentile(boot_r, 2.5) > 0), "cfg": (name(jb), name(ja))})
    pr(f"{'бюджет':>7} | {'мкс b/a':>11} | {'полнота b/a':>13} {'adapt-base [CI]':>24} {'знач.':>5} | "
       f"{'MRR b/a':>13} {'adapt-base [CI]':>24}")
    for B in budgets:
        v = rows.get(B)
        if not v or len(v) < len(args.seeds):
            continue
        g = lambda key: np.mean([x[key] for x in v], axis=0)
        cir, cim = g("dr_ci"), g("dm_ci")
        pr(f"{B:>7.0f} | {g('lat_b'):>5.0f}/{g('lat_a'):<5.0f} | {g('rec_b'):.3f}/{g('rec_a'):.3f} "
           f"{g('dr'):>+8.4f} [{cir[0]:+.4f},{cir[1]:+.4f}] {sum(x['sig_r'] for x in v):>2}/{len(v)} | "
           f"{g('mrr_b'):.4f}/{g('mrr_a'):.4f} {g('dm'):>+8.4f} [{cim[0]:+.4f},{cim[1]:+.4f}]")
    pr("   знач. — в скольких seed нижняя граница CI разности полноты > 0 (адаптивный значимо точнее при том же бюджете).")

    # 3. обратная сторона: задержка при равной полноте
    pr("\n3. Задержка при равной полноте (фронт на всех запросах, линейная интерполяция)")
    pr(f"{'полнота':>8} | {'обычный, мкс':>12} {'адаптивный, мкс':>15} {'ratio':>6}")
    for target in (0.85, 0.90, 0.95, 0.97, 0.98, 0.99):
        res = []
        for lab in ("base", "adapt"):
            fl, fq = fronts[lab, "rec"]
            res.append(float(np.interp(target, fq, fl)) if fq[0] <= target <= fq[-1] else np.nan)
        pr(f"{target:>8.2f} | {res[0]:>12.0f} {res[1]:>15.0f} {res[1] / res[0]:>6.2f}")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    rows_csv = ["method,config,latency_us,speedup,recall,mrr"]
    for j in range(len(kind)):
        rows_csv.append(f"{kind[j]},{name(j)},{lat_m[j]:.1f},{lat_m[ref] / lat_m[j]:.3f},{rec_m[j]:.4f},{mrr_m[j]:.4f}")
    with open(args.report.rsplit(".", 1)[0] + ".csv", "w", encoding="utf-8") as f:
        f.write("\n".join(rows_csv) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (key, vals, ylab) in zip(axes, (("rec", rec_m, "полнота@10 к точному поиску"),
                                                 ("mrr", mrr_m, "MRR@10"))):
            for lab, cols, col, title in (("base", base, "#888888", "SEISMIC: подбор query_cut, heap_factor"),
                                          ("adapt", adapt, "#1f6feb", "адаптивный SEISMIC")):
                ax.scatter(lat_m[ref] / lat_m[cols], vals[cols], s=6, color=col, alpha=0.25)
                fl, fq = fronts[lab, key]
                ax.step(lat_m[ref] / fl, fq, where="post", color=col, lw=2, label=title)
            if key == "mrr":
                ax.axhline(exact_mrr, color="#444", ls=":", lw=1, label="точный поиск")
            ax.set_xscale("log")
            ax.set_xlabel("ускорение относительно самой точной настройки SEISMIC")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            lo = np.percentile(vals, 5)
            ax.set_ylim(lo, vals.max() + (vals.max() - lo) * 0.05)
        axes[0].legend(fontsize=8, loc="lower left")
        fig.suptitle(f"{args.name}: скорость против качества")
        fig.tight_layout()
        png = args.report.rsplit(".", 1)[0] + ".png"
        fig.savefig(png, dpi=150)
        print("график:", png)
    except ImportError:
        print("matplotlib нет — график не построен (точки фронтов — в CSV)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True, help="npz с замерами")
    ap.add_argument("--report", help="текстовый отчёт (по умолчанию рядом с --out, .txt)")
    ap.add_argument("--name", default="")
    ap.add_argument("--n-postings", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    args.report = args.report or os.path.splitext(args.out)[0] + ".txt"
    if not args.analyze_only:
        docs = sp.load_npz(args.docs).tocsr()
        docs.sort_indices()
        queries = sp.load_npz(args.queries).tocsr()
        queries.sort_indices()
        measure(args, docs, queries)
    analyze(args)


if __name__ == "__main__":
    main()
