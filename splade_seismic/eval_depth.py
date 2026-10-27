"""Диагностика выбора cut и адаптивная ГЛУБИНА обхода списков (прокси-симулятор, как eval_cut_v2.py).

Часть 1 — диагностика гипотезы «на большом корпусе выбирать число списков бесполезно»:
  * какая доля главных термов запроса имеет список длиннее lam (то есть обрезанный ровно до lam);
  * какой cut нужен запросам (распределение оракульного cut).

Часть 2 — вместо «обойти cut списков целиком» постинги всех термов запроса обходятся в порядке убывания
вклада q_i * w(doc, i) (impact order), и останавливаемся после N постингов. N — непрерывная ручка глубины.
Кандидаты = уникальные документы среди первых N постингов, оцениваются точно (как через forward index).

Методы:
  fixed_cut — одинаковое число списков (как раньше; базовая линия);
  fixed_N   — одинаковый бюджет постингов N для всех запросов (эффект самого порядка обхода);
  relthr    — останов, когда вклад очередного постинга < theta * (самый большой вклад запроса);
  lin_N     — log N предсказывается линейной моделью по признакам запроса (обучение на tune);
  kth_gain  — во время поиска: стоп, когда k-й скор кандидатов вырос меньше чем на eps за последнее удвоение N;
  sat_bound — во время поиска: стоп, когда k-й скор >= f * (верхняя оценка скора ещё не встреченного документа);
              при f = 1 останов безопасен: ни один пропущенный документ не обгонит найденный top-k;
  probe_*   — обходим P постингов, по увиденному (скоры кандидатов, падение вклада, оценка) предсказываем log N
              (probe_lin — линейная модель, probe_gbr — бустинг);
  oracle_N  — наименьший N с полнотой запроса >= порога (потолок);
  oracle_cut— то же для числа списков (потолок для старой постановки).
ratio_cut = стоимость / стоимость fixed_cut при той же полноте; ratio_N — то же относительно fixed_N.

  python eval_depth_2.py --docs docs.npz --queries queries.npz --lam 1000
"""
import argparse

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_adaptive_cut import make_synthetic
from eval_cut_v2 import bootstrap_ratio, coord_stats, fixed_interp, query_features


class QueryTrace:
    """Всё, что нужно, чтобы для запроса быстро считать стоимость и полноту при любом N или cut."""
    __slots__ = ("imp", "first_sorted", "fp_top", "rmin_sorted", "rmin_top", "T", "n_terms",
                 "kth", "top1", "bound")  # последние три — на контрольных точках CK (сигналы «во время поиска»)

    def cost_N(self, n):
        return np.searchsorted(self.first_sorted, n, side="left")

    def recall_N(self, n, k):
        return np.searchsorted(self.fp_top, n, side="left") / k

    def cost_cut(self, m):
        return np.searchsorted(self.rmin_sorted, m, side="left")

    def recall_cut(self, m, k):
        return np.searchsorted(self.rmin_top, m, side="left") / k


def build_trace(docs_csc, lists, idx, val, k, max_terms=None, ck=None, maxw=None, thr=None):
    """idx, val — ВСЕ координаты запроса по убыванию веса: точный top-k считается полным запросом,
    а в поиске участвуют только первые max_terms.
    ck — контрольные точки N; на них запоминаются k-й и лучший скор среди уже найденных кандидатов
    и верхняя оценка скора документа, который ещё ни разу не встретился (нужны maxw и thr из coord_stats)."""
    scores = np.asarray(docs_csc[:, idx] @ val).ravel()
    exact = np.argpartition(-scores, k)[:k]
    idx_all, val_all = idx, val
    if max_terms is not None:
        idx, val = idx[:max_terms], val[:max_terms]
    ids, imps, ranks = [], [], []
    for r, (j, q) in enumerate(zip(idx, val)):
        pid, pw = lists[j]
        ids.append(pid)
        imps.append(q * pw)
        ranks.append(np.full(len(pid), r, dtype=np.int32))
    ids, imps, ranks = np.concatenate(ids), np.concatenate(imps), np.concatenate(ranks)

    order = np.argsort(-imps, kind="stable")
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    uniq, inv = np.unique(ids, return_inverse=True)
    first = np.full(len(uniq), np.iinfo(np.int64).max)
    np.minimum.at(first, inv, pos)
    rmin = np.full(len(uniq), np.iinfo(np.int32).max, dtype=np.int64)
    np.minimum.at(rmin, inv, ranks)

    p = np.searchsorted(uniq, exact)
    p = np.clip(p, 0, len(uniq) - 1)
    hit = uniq[p] == exact
    inf = np.iinfo(np.int64).max
    t = QueryTrace()
    t.imp = imps[order].astype(np.float32)
    t.first_sorted = np.sort(first)
    t.fp_top = np.sort(np.where(hit, first[p], inf))
    t.rmin_sorted = np.sort(rmin)
    t.rmin_top = np.sort(np.where(hit, rmin[p], inf))
    t.T = len(order)
    t.n_terms = len(idx)
    if ck is not None:
        ordf = np.argsort(first, kind="stable")
        s_ord = scores[uniq][ordf]                 # точные скоры кандидатов в порядке их появления
        cummax = np.maximum.accumulate(s_ord)
        imp_sorted = imps[order]
        n_at = np.minimum(ck, t.T)
        cnt = np.searchsorted(t.first_sorted, n_at, side="left")
        t.kth = np.zeros(len(ck))
        t.top1 = np.zeros(len(ck))
        for i, c in enumerate(cnt):
            if c >= k:
                t.kth[i] = np.partition(s_ord[:c], c - k)[c - k]
            if c > 0:
                t.top1[i] = cummax[c - 1]
        # Документ, ещё не встреченный после N постингов: по каждому терму поиска его вклад не больше
        # следующего вклада в порядке обхода (и не больше q*maxw), а если его нет в урезанном списке —
        # не больше q*порог обрезки; термы вне поиска дают не больше q*maxw.
        q_max = val * maxw[idx]
        q_thr = val * thr[idx]
        extra = float(np.sum(val_all[len(idx):] * maxw[idx_all[len(idx):]]))
        nxt = np.where(n_at < t.T, imp_sorted[np.minimum(n_at, t.T - 1)], 0.0)
        t.bound = np.array([np.maximum(np.minimum(x, q_max), q_thr).sum() for x in nxt]) + extra
    return t


def weighted_lists(docs_csc, cols, lam):
    lists = {}
    for j in cols:
        s, e = docs_csc.indptr[j], docs_csc.indptr[j + 1]
        pid, pw = docs_csc.indices[s:e], docs_csc.data[s:e]
        if len(pid) > lam:
            keep = np.argpartition(-pw, lam)[:lam]
            pid, pw = pid[keep], pw[keep]
        lists[j] = (pid, pw.astype(np.float64))
    return lists


def evaluate_N(traces, n_per_query, k):
    n = np.maximum(1, np.minimum(np.floor(n_per_query), [t.T for t in traces])).astype(np.int64)
    rec = np.array([t.recall_N(x, k) for t, x in zip(traces, n)])
    cost = np.array([t.cost_N(x) for t, x in zip(traces, n)], dtype=float)
    return rec, cost


def oracle_min(sorted_pos, tau, k, cap):
    """Наименьшее значение ручки (N или cut), при котором полнота запроса >= tau."""
    need = int(np.ceil(tau * k - 1e-9))
    if need == 0:
        return 1
    v = sorted_pos[need - 1]
    return cap if v == np.iinfo(np.int64).max else int(v) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--docs")
    ap.add_argument("--queries")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lam", type=int, default=1000)
    ap.add_argument("--max-terms", type=int, default=50, help="сколько главных термов запроса участвуют в поиске")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.97])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tune-frac", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-docs", type=int, default=20000)
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--n-queries", type=int, default=400)
    args = ap.parse_args()
    k = args.k

    if args.synthetic:
        docs, queries = make_synthetic(0, args.n_docs, args.vocab, args.n_queries)
    else:
        docs, queries = sp.load_npz(args.docs).tocsr(), sp.load_npz(args.queries).tocsr()
    docs_csc = docs.tocsc()
    docs_csc.sort_indices()
    queries.sort_indices()
    n_docs = docs.shape[0]
    df, maxw, thr = coord_stats(docs, args.lam)

    q_items = []
    for qi in range(queries.shape[0]):
        s, e = queries.indptr[qi], queries.indptr[qi + 1]
        idx, val = queries.indices[s:e], queries.data[s:e]
        o = np.argsort(-val)
        q_items.append((idx[o], val[o].astype(np.float64)))
    lists = weighted_lists(docs_csc, np.unique(np.concatenate([i[:args.max_terms] for i, _ in q_items])), args.lam)

    t_upper = max(sum(len(lists[j][0]) for j in idx[:args.max_terms]) for idx, _ in q_items)
    CK = np.unique(np.round(2 ** (np.arange(0, 4 * np.log2(max(t_upper, 2)) + 1) / 4)).astype(np.int64))
    traces, feats = [], []
    for idx, val in q_items:
        t = build_trace(docs_csc, lists, idx, val, k, args.max_terms, CK, maxw, thr)
        traces.append(t)
        imp = t.imp
        at = lambda i: np.log(max(imp[min(i, len(imp) - 1)], 1e-12) / max(imp[0], 1e-12))
        feats.append(np.concatenate([query_features(idx[:args.max_terms], val[:args.max_terms], df, maxw, thr, n_docs),
                                     [np.log(t.T), np.log(max(imp[0], 1e-12)), at(10), at(100), at(1000)]]))
    feats = np.array(feats)
    n_q = len(traces)
    max_cut = args.max_terms

    # ---------------- часть 1: диагностика
    top5 = np.concatenate([i[:5] for i, _ in q_items])
    print(f"docs={n_docs} queries={n_q} lam={args.lam} k={k}; ненулевых: документ {docs.nnz / n_docs:.1f}, "
          f"запрос {queries.nnz / n_q:.1f}")
    print("\n=== Диагностика ===")
    print(f"доля 5 главных термов запроса, чей список длиннее lam (обрезан ровно до lam): "
          f"{np.mean(df[top5] > args.lam):.2f}")
    print(f"медиана длины полного списка (df) у главного терма: {np.median(df[[i[0] for i, _ in q_items]]):.0f}")
    for tau in (0.9, 1.0):
        oc = np.array([oracle_min(t.rmin_top, tau, k, max_cut + 1) for t in traces])
        bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, max_cut), (max_cut + 1, max_cut + 1)]
        txt = ", ".join(f"{a}" + (f"–{b}" if b != a else "") + f": {np.mean((oc >= a) & (oc <= b)):.2f}"
                        if a <= max_cut else f"недостижимо: {np.mean(oc > max_cut):.2f}" for a, b in bins)
        on = np.array([oracle_min(t.fp_top, tau, k, t.T) for t in traces])
        print(f"полнота запроса >= {tau}: оракульный cut — {txt}")
        print(f"    оракульный N (постингов): p10 {np.percentile(on, 10):.0f}, медиана {np.median(on):.0f}, "
              f"p90 {np.percentile(on, 90):.0f}  (для сравнения lam = {args.lam})")

    # кривые для базовых линий
    Tmax = max(t.T for t in traces)
    grid = np.unique(np.round(np.geomspace(1, Tmax, 120)).astype(np.int64))
    Rc = np.array([[t.recall_cut(m, k) for m in range(1, max_cut + 1)] for t in traces])
    Cc = np.array([[t.cost_cut(m) for m in range(1, max_cut + 1)] for t in traces], dtype=float)
    RN = np.array([[t.recall_N(min(n, t.T), k) for n in grid] for t in traces])
    CN = np.array([[t.cost_N(min(n, t.T)) for n in grid] for t in traces], dtype=float)

    # ---------------- часть 2: методы
    methods = ["fixed_cut", "fixed_N", "relthr", "lin_N", "kth_gain", "sat_bound", "probe_lin", "probe_gbr",
               "oracle_N", "oracle_cut"]
    KTH = np.array([t.kth for t in traces])
    TOP1 = np.array([t.top1 for t in traces])
    BND = np.array([t.bound for t in traces])
    Tq = np.array([t.T for t in traces])
    G = len(CK)

    def stop_to_N(cond):
        """Первая контрольная точка, где выполнено условие; если нигде — обходим всё (N = T)."""
        has = cond.any(1)
        i = cond.argmax(1)
        return np.where(has, np.minimum(CK[i], Tq), Tq)

    # признаки после пробного обхода P постингов (для probe_*)
    def probe_feats(p):
        i = min(int(np.searchsorted(CK, p)), G - 1)
        j = max(i - 4, 0)  # примерно P/2
        imp_p = np.array([t.imp[min(CK[i], t.T - 1)] / max(t.imp[0], 1e-12) for t in traces])
        cnt_p = np.array([t.cost_N(min(CK[i], t.T)) for t in traces], dtype=float)
        kth, top1, bnd = KTH[:, i], TOP1[:, i], BND[:, i]
        growth = np.where(kth > 0, (kth - KTH[:, j]) / np.maximum(kth, 1e-12), 1.0)
        f = np.column_stack([np.log(np.maximum(imp_p, 1e-12)), np.log1p(kth), kth / np.maximum(top1, 1e-12),
                             cnt_p / CK[i], np.log(np.maximum(bnd, 1e-12) / np.maximum(kth, 1e-12)), growth])
        return np.hstack([feats, f]), CK[i]
    summary = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_q)
        n_tune = int(round(args.tune_frac * n_q))
        tune, test = np.sort(perm[:n_tune]), np.sort(perm[n_tune:])

        cfg = {m: [] for m in methods}  # name -> [(params, recall_q, cost_q)]
        for j, m in enumerate(range(1, max_cut + 1)):
            cfg["fixed_cut"].append((f"cut={m}", Rc[:, j], Cc[:, j]))
        for j, n in enumerate(grid):
            cfg["fixed_N"].append((f"N={n}", RN[:, j], CN[:, j]))
        for th in np.geomspace(1e-3, 1.0, 80):
            n = np.array([np.searchsorted(-t.imp, -th * t.imp[0], side="right") for t in traces])
            cfg["relthr"].append((f"theta={th:.4f}", *evaluate_N(traces, n, k)))
        for tau in np.linspace(0.5, 1.0, 51):
            n = np.array([oracle_min(t.fp_top, tau, k, t.T) for t in traces])
            cfg["oracle_N"].append((f"recall_q>={tau:.2f}", *evaluate_N(traces, n, k)))
            m = np.array([min(oracle_min(t.rmin_top, tau, k, max_cut), max_cut) for t in traces])
            cfg["oracle_cut"].append((f"recall_q>={tau:.2f}", Rc[np.arange(n_q), m - 1], Cc[np.arange(n_q), m - 1]))
        make_lin = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        for tau in (0.7, 0.8, 0.9, 1.0):
            y = np.log([oracle_min(t.fp_top, tau, k, t.T) for t in traces])
            pred = np.zeros(n_q)
            for tr, va in KFold(5, shuffle=True, random_state=seed).split(tune):
                pred[tune[va]] = make_lin().fit(feats[tune[tr]], y[tune[tr]]).predict(feats[tune[va]])
            pred[test] = make_lin().fit(feats[tune], y[tune]).predict(feats[test])
            for d in np.linspace(-1.5, 3.0, 46):
                cfg["lin_N"].append((f"tau={tau},d={d:+.1f}", *evaluate_N(traces, np.exp(pred + d), k)))

        # kth_gain: k-й скор почти не вырос за последнее удвоение N
        gain = np.where(KTH[:, 4:] > 0, (KTH[:, 4:] - KTH[:, :-4]) / np.maximum(KTH[:, 4:], 1e-12), np.inf)
        for eps in (0.0, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2):
            for nmin in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
                cond = np.zeros((n_q, G), dtype=bool)
                cond[:, 4:] = (gain <= eps) & (CK[4:] >= nmin)[None, :]
                cfg["kth_gain"].append((f"eps={eps},nmin={nmin}", *evaluate_N(traces, stop_to_N(cond), k)))
        # sat_bound: даже лучший возможный ещё не встреченный документ не обгонит k-й (с запасом f)
        for f_ in np.geomspace(0.02, 1.0, 40):
            cfg["sat_bound"].append((f"f={f_:.3f}", *evaluate_N(traces, stop_to_N(KTH >= f_ * BND), k)))
        # probe: обходим P постингов, по увиденному предсказываем log N
        make_gbr = lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                                     subsample=0.8, random_state=seed)
        for p_ in (64, 256):
            xp, p_real = probe_feats(p_)
            for name, make in (("probe_lin", make_lin), ("probe_gbr", make_gbr)):
                for tau in (0.7, 0.8, 0.9, 1.0):
                    y = np.log([oracle_min(t.fp_top, tau, k, t.T) for t in traces])
                    pred = np.zeros(n_q)
                    for tr, va in KFold(5, shuffle=True, random_state=seed).split(tune):
                        pred[tune[va]] = make().fit(xp[tune[tr]], y[tune[tr]]).predict(xp[tune[va]])
                    pred[test] = make().fit(xp[tune], y[tune]).predict(xp[test])
                    for d in np.linspace(-1.5, 3.0, 31):
                        n = np.maximum(p_real, np.exp(pred + d))
                        cfg[name].append((f"P={p_real},tau={tau},d={d:+.1f}", *evaluate_N(traces, n, k)))

        fc_r, fc_c = Rc[test], Cc[test]
        fn_r, fn_c = RN[test], CN[test]
        for target in args.targets:
            for name in methods:
                part = test if name.startswith("oracle") else tune
                ok = [c for c in cfg[name] if c[1][part].mean() >= target]
                if not ok:
                    summary.setdefault((target, name), []).append(None)
                    continue
                params, r, c = min(ok, key=lambda c: c[2][part].mean())
                rt, ct = r[test], c[test]
                base_c = fixed_interp(fc_r.mean(0), fc_c.mean(0), rt.mean())
                base_n = fixed_interp(fn_r.mean(0), fn_c.mean(0), rt.mean())
                p95_c = fixed_interp(fc_r.mean(0), np.percentile(fc_c, 95, axis=0), rt.mean())
                _, hi_c = bootstrap_ratio(rt, ct, fc_r, fc_c, args.n_boot, rng)
                _, hi_n = bootstrap_ratio(rt, ct, fn_r, fn_c, args.n_boot, rng)
                summary.setdefault((target, name), []).append({
                    "recall": rt.mean(), "cost": ct.mean(), "p95": np.percentile(ct, 95),
                    "ratio_cut": ct.mean() / base_c if base_c else np.nan,
                    "ratio_N": ct.mean() / base_n if base_n else np.nan,
                    "p95r_cut": np.percentile(ct, 95) / p95_c if p95_c else np.nan,
                    "hi_c": hi_c, "hi_n": hi_n})

    print(f"\n=== Методы (seeds {args.seeds}; параметры выбраны на tune по средней стоимости, числа на test) ===")
    print("ratio_cut / ratio_N — стоимость относительно fixed_cut / fixed_N при той же полноте (среднее по seed);")
    print("CI<1 — в скольких seed верхняя граница 95% bootstrap-интервала меньше 1; p95r_cut — хвост относительно fixed_cut\n")
    print(f"{'target':>6} | {'метод':<10} {'recall':>6} {'cost':>7} {'p95':>7} {'ratio_cut':>9} {'CI<1':>5} "
          f"{'ratio_N':>7} {'CI<1':>5} {'p95r_cut':>8}")
    ci = lambda hi, is_self: "—" if is_self else f"{int(np.sum(hi < 1 - 1e-9))}/{len(hi)}"
    for target in args.targets:
        for name in methods:
            vals = summary[(target, name)]
            if any(v is None for v in vals):
                print(f"{target:>6.2f} | {name:<10} недостижимо хотя бы в одном seed")
                continue
            g = lambda key: np.array([v[key] for v in vals], dtype=float)
            print(f"{target:>6.2f} | {name:<10} {g('recall').mean():>6.3f} {g('cost').mean():>7.0f} "
                  f"{g('p95').mean():>7.0f} {np.nanmean(g('ratio_cut')):>9.2f} "
                  f"{ci(g('hi_c'), name == 'fixed_cut'):>5} {np.nanmean(g('ratio_N')):>7.2f} "
                  f"{ci(g('hi_n'), name == 'fixed_N'):>5} {np.nanmean(g('p95r_cut')):>8.2f}")
        print()
    print("Оракулы смотрят на ответ и нереализуемы; при k=10 их строки для высокой цели искажены дискретностью.")


if __name__ == "__main__":
    main()
