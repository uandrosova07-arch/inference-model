"""Сравнение способов выбора cut для каждого запроса (прокси-симулятор, как eval_adaptive_cut.py).

Модель поиска: списки урезаны до top-lam документов по весу; координаты запроса обходятся по убыванию веса;
после m координат кандидаты = объединение их списков, кандидаты оцениваются точно полным запросом.
Полнота = доля точного top-k среди кандидатов; стоимость = число кандидатов.

Методы:
  fixed     — одинаковый cut для всех запросов;
  gamma     — наименьший m, при котором top-m координат несут долю gamma L1-массы запроса;
  stable    — остановка, когда top-k среди кандидатов не менялся s шагов подряд;
  bound     — остановка, когда k-й скор кандидатов >= factor * верхняя оценка скора ещё не найденного документа;
  budget    — обходить координаты, пока сумма длин их урезанных списков < B (решение до поиска, по индексу);
  budget_c  — обходить, пока реальное число уникальных кандидатов < B (решение во время поиска);
  learned   — cut предсказывается бустингом по признакам запроса (обучение только на tune-части);
  linear    — то же, но гребневая регрессия (линейная формула) по всем признакам;
  lin_top3  — линейная регрессия только по трём самым важным для бустинга признакам;
  lin+bud   — cut от linear, но не больше точки, где сумма длин списков достигает бюджета B;
  oracle    — наименьший cut с полнотой запроса >= порога (потолок, нереализуем).

Честность сравнения:
  * запросы делятся на tune и test; параметры каждого метода выбираются на tune (самая дешёвая
    конфигурация с полнотой >= цели), отчёт — на test;
  * стоимость метода сравнивается со стоимостью fixed cut, линейно интерполированной на ТУ ЖЕ полноту;
  * 95% bootstrap-интервал для отношения стоимостей (по запросам test).

Запуск:
  python eval_cut_v2.py --synthetic
  python eval_cut_v2.py --docs docs.npz --queries queries.npz --lam 500 --max-cut 50   (seeds 0 1 2 по умолчанию)
"""
import argparse
import csv

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_adaptive_cut import build_postings, make_synthetic


# ---------------------------------------------------------------- трассы поиска

def coord_stats(docs: sp.csr_matrix, lam: int):
    """Для каждой координаты: df (полная длина списка), максимальный вес, порог обрезки
    (вес lam-го элемента; 0, если список не длиннее lam)."""
    csc = docs.tocsc()
    v = csc.shape[1]
    df = np.diff(csc.indptr)
    maxw = np.zeros(v)
    thr = np.zeros(v)
    for j in range(v):
        w = csc.data[csc.indptr[j]:csc.indptr[j + 1]]
        if len(w):
            maxw[j] = w.max()
            if len(w) > lam:
                thr[j] = np.partition(w, len(w) - lam)[len(w) - lam]
    return df, maxw, thr


def trace_query(docs, postings, idx, val, maxw, thr, k, max_cut):
    """Пошаговая трасса для одного запроса (idx, val отсортированы по убыванию val).

    Возвращает массивы длины max_cut (значение после m+1 координат):
      recall, cost, changed (изменился ли top-k кандидатов на этом шаге), kth (k-й скор кандидатов), bound.
    """
    n = docs.shape[0]
    q = sp.csr_matrix((val, (np.zeros(len(idx), dtype=int), idx)), shape=(1, docs.shape[1]))
    scores = (docs @ q.T).toarray().ravel()
    exact = np.argpartition(-scores, k)[:k]

    # bound_m: верхняя оценка скора документа, не попавшего ни в один из первых m списков.
    # Для пройденных координат его вес <= порога обрезки, для остальных <= максимального веса.
    pref_thr = np.concatenate([[0.0], np.cumsum(val * thr[idx])])
    suf_max = np.concatenate([np.cumsum((val * maxw[idx])[::-1])[::-1], [0.0]])

    recall = np.zeros(max_cut)
    cost = np.zeros(max_cut)
    changed = np.zeros(max_cut, dtype=bool)
    kth = np.zeros(max_cut)
    bound = np.zeros(max_cut)

    mask = np.zeros(n, dtype=bool)
    prev_top = frozenset()
    for m in range(max_cut):
        if m < len(idx):
            mask[postings[idx[m]]] = True
        mm = min(m + 1, len(idx))
        cand = np.flatnonzero(mask)
        recall[m] = mask[exact].mean()
        cost[m] = len(cand)
        if len(cand):
            cs = scores[cand]
            if len(cand) > k:
                top_pos = np.argpartition(-cs, k - 1)[:k]
                kth[m] = cs[top_pos].min()
            else:
                top_pos = np.arange(len(cand))
                kth[m] = 0.0 if len(cand) < k else cs.min()
            top = frozenset(cand[top_pos].tolist())
        else:
            top = frozenset()
        changed[m] = top != prev_top
        prev_top = top
        bound[m] = pref_thr[mm] + suf_max[mm]
    return recall, cost, changed, kth, bound


FEATURE_NAMES = (["nnz", "w1", "w1/w2", "entropy", "mass_top3", "mass_top5", "l1"]
                 + [f"logdf_t{i}" for i in range(1, 6)]          # log(df/N) i-го по весу терма запроса
                 + [f"maxcontrib_t{i}" for i in range(1, 6)]     # q_i * максимальный вес терма в корпусе
                 + [f"thrcontrib_t{i}" for i in range(1, 6)])    # q_i * вес на пороге обрезки списка


def query_features(idx, val, df, maxw, thr, n_docs):
    v = np.asarray(val, dtype=np.float64)
    p = v / v.sum()
    cs = np.cumsum(v) / v.sum()
    top = idx[:5]
    pad = lambda a: np.pad(a, (0, 5 - len(a)), constant_values=0.0)
    return np.concatenate([
        [len(v), v[0], v[0] / v[1] if len(v) > 1 else 1.0, -(p * np.log(p)).sum(),
         cs[min(2, len(v) - 1)], cs[min(4, len(v) - 1)], v.sum()],
        pad(np.log1p(df[top]) - np.log(n_docs)),       # насколько «частые» главные термы
        pad(maxw[top] * v[:5]),                          # максимально возможный вклад главных термов
        pad(thr[top] * v[:5]),                           # вклад на пороге обрезки
    ])


# ---------------------------------------------------------------- правила остановки

def first_true(cond):
    """Индекс (0-based) первого True в каждой строке; если нет — последний индекс."""
    has = cond.any(axis=1)
    return np.where(has, cond.argmax(axis=1), cond.shape[1] - 1)


def stop_stable(changed, s, cmin):
    """Остановиться на шаге m, если на шагах m-s+1..m top-k не менялся и m+1 >= cmin."""
    n, t = changed.shape
    unchanged = ~changed
    run = np.zeros_like(changed, dtype=int)
    for m in range(t):
        run[:, m] = np.where(unchanged[:, m], (run[:, m - 1] if m else 0) + 1, 0)
    cond = (run >= s) & (np.arange(t)[None, :] + 1 >= cmin)
    return first_true(cond)


def stop_bound(kth, bound, factor):
    return first_true(kth >= factor * bound)


# ---------------------------------------------------------------- оценка

def evaluate(stop_idx, recall, cost):
    r = np.arange(len(stop_idx))
    return recall[r, stop_idx], cost[r, stop_idx]


def fixed_interp(fix_rec, fix_cost, target):
    """Стоимость fixed cut на средней полноте target (линейная интерполяция по кривой); nan вне диапазона."""
    if target < fix_rec[0] or target > fix_rec[-1]:
        return np.nan
    # кривая неубывающая по полноте; убираем повторы, чтобы interp был корректен
    keep = np.concatenate([[True], np.diff(fix_rec) > 1e-12])
    return float(np.interp(target, fix_rec[keep], fix_cost[keep]))


def bootstrap_ratio(rec_q, cost_q, fix_rec_q, fix_cost_q, n_boot, rng):
    """95% интервал для cost_method / cost_fixed_interp при той же полноте."""
    n = len(rec_q)
    out = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        fr, fc = fix_rec_q[s].mean(0), fix_cost_q[s].mean(0)
        base = fixed_interp(fr, fc, rec_q[s].mean())
        if np.isfinite(base) and base > 0:
            out.append(cost_q[s].mean() / base)
    if len(out) < n_boot // 2:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--docs")
    ap.add_argument("--queries")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lam", type=int, default=200)
    ap.add_argument("--max-cut", type=int, default=30)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.97])
    ap.add_argument("--tune-frac", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="разбиения запросов на tune/test; результаты усредняются")
    ap.add_argument("--select-by", choices=["cost", "cost_p95"], default="cost",
                    help="что минимизировать при выборе параметров на tune: среднюю стоимость или её p95")
    ap.add_argument("--verbose", action="store_true", help="печатать подробную таблицу для каждого seed")
    ap.add_argument("--n-docs", type=int, default=20000)
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--curves-csv", help="записать все конфигурации (tune/test) в CSV для графиков")
    args = ap.parse_args()

    if args.synthetic:
        docs, queries = make_synthetic(args.seeds[0], args.n_docs, args.vocab, args.n_queries)
    else:
        docs, queries = sp.load_npz(args.docs).tocsr(), sp.load_npz(args.queries).tocsr()
    docs.sort_indices()
    queries.sort_indices()

    postings = build_postings(docs, args.lam)
    df, maxw, thr = coord_stats(docs, args.lam)

    n_q, t = queries.shape[0], args.max_cut
    R, C, CH, KTH, B = (np.zeros((n_q, t)) for _ in range(5))
    CH = CH.astype(bool)
    qvals, feats = [], []
    LISTLEN = np.zeros((n_q, t))  # накопленная сумма длин (урезанных) списков первых m координат запроса
    for qi in range(n_q):
        s, e = queries.indptr[qi], queries.indptr[qi + 1]
        idx, val = queries.indices[s:e], queries.data[s:e]
        o = np.argsort(-val)
        idx, val = idx[o], val[o]
        qvals.append(val)
        feats.append(query_features(idx, val, df, maxw, thr, docs.shape[0]))
        R[qi], C[qi], CH[qi], KTH[qi], B[qi] = trace_query(docs, postings, idx, val, maxw, thr, args.k, t)
        lens = np.array([len(postings[j]) for j in idx[:t]], dtype=float)
        LISTLEN[qi] = np.cumsum(np.pad(lens, (0, t - len(lens))))  # после последней координаты сумма не растёт
    feats = np.array(feats)

    stats = dict(R=R, C=C, CH=CH, KTH=KTH, B=B, LISTLEN=LISTLEN, feats=feats, qvals=qvals, t=t, n_q=n_q)
    all_rows, summary = [], {}
    for i, seed in enumerate(args.seeds):
        verbose = args.verbose or len(args.seeds) == 1
        if verbose:
            print(f"\n================ seed {seed} ================")
        res, rows = run_seed(seed, args, stats, verbose=verbose, importance=(i == 0))
        all_rows += rows
        for key, val in res.items():
            summary.setdefault(key, []).append(val)

    if args.curves_csv:
        with open(args.curves_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)

    print(f"\ndocs={docs.shape[0]} queries={n_q} k={args.k} lam={args.lam} max_cut={t} seeds={args.seeds}")
    print(f"Сводка по разбиениям (параметры выбраны на tune по минимуму {args.select_by}, числа на test).")
    print("ratio  = стоимость / стоимость fixed cut при той же полноте: среднее по seed, в скобках min–max;")
    print("CI<1   = в скольких seed верхняя граница 95% bootstrap-интервала ratio меньше 1;")
    print("p95r   = p95 стоимости / p95 fixed cut при той же полноте (интерполяция), среднее по seed.\n")
    print(f"{'target':>6} | {'метод':<9} {'recall':>6} {'cost':>7} {'p95':>7} {'ratio':>16} {'CI<1':>5} {'p95r':>6}")
    for target in args.targets:
        for name in METHODS:
            vals = summary.get((target, name))
            if not vals:
                continue
            if any(v is None for v in vals):
                print(f"{target:>6.2f} | {name:<9} недостижимо хотя бы в одном seed")
                continue
            g = lambda k: np.array([v[k] for v in vals], dtype=float)
            r = g("ratio")
            ci_ok = int(np.sum(g("hi") < 1.0))
            print(f"{target:>6.2f} | {name:<9} {g('recall').mean():>6.3f} {g('cost').mean():>7.0f} "
                  f"{g('p95').mean():>7.0f} {np.nanmean(r):>6.2f} ({np.nanmin(r):.2f}–{np.nanmax(r):.2f}) "
                  f"{ci_ok:>2}/{len(vals):<2} {np.nanmean(g('p95r')):>6.2f}")
        print()
    print("Строки oracle при цели 0.97 искажены дискретностью полноты запроса (k=10): не учитывайте их.")


METHODS = ["fixed", "gamma", "stable", "bound", "budget", "budget_c", "learned", "linear", "lin_top3",
           "lin+bud", "oracle"]


def run_seed(seed, args, st_, verbose, importance):
    R, C, CH, KTH, B, LISTLEN = st_["R"], st_["C"], st_["CH"], st_["KTH"], st_["B"], st_["LISTLEN"]
    feats, qvals, t, n_q = st_["feats"], st_["qvals"], st_["t"], st_["n_q"]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_q)
    n_tune = int(round(args.tune_frac * n_q))
    tune, test = np.sort(perm[:n_tune]), np.sort(perm[n_tune:])

    # --- конфигурации: name -> [(params_str, stop_idx для всех запросов)]
    configs = {m: [] for m in METHODS}
    for m in range(1, t + 1):
        configs["fixed"].append((f"cut={m}", np.full(n_q, m - 1)))
    for g in np.linspace(0.05, 1.0, 96):
        ms = []
        for v in qvals:
            c = np.cumsum(v) / v.sum()
            ms.append(min(int(np.searchsorted(c, g)) + 1, t))
        configs["gamma"].append((f"gamma={g:.2f}", np.array(ms) - 1))
    for s_ in range(1, 16):
        for cmin in (1, 2, 3, 4, 6, 8):
            configs["stable"].append((f"s={s_},cmin={cmin}", stop_stable(CH, s_, cmin)))
    for f in np.round(np.geomspace(0.02, 2.0, 60), 4):
        configs["bound"].append((f"factor={f}", stop_bound(KTH, B, f)))
    # budget: идём по координатам, пока сумма длин их списков не достигнет B (известно ДО поиска, из индекса);
    # budget_c: пока реальное число уникальных кандидатов не достигнет B (наблюдается ВО ВРЕМЯ поиска).
    b_grid = np.unique(np.round(np.geomspace(10, max(LISTLEN.max(), 20), 120)))
    budget_stop = {}
    for b in b_grid:
        budget_stop[b] = first_true(LISTLEN >= b)
        configs["budget"].append((f"B={int(b)}", budget_stop[b]))
        configs["budget_c"].append((f"B={int(b)}", first_true(C >= b)))
    for thr_q in np.linspace(0.5, 1.0, 51):
        ms = np.where((R >= thr_q - 1e-12).any(1), (R >= thr_q - 1e-12).argmax(1), t - 1)
        configs["oracle"].append((f"recall_q>={thr_q:.2f}", ms))

    # learned / linear / lin_top3: цель — оракульный cut (порог tau на запрос); предсказание сдвигается на delta.
    # Предсказания для tune — out-of-fold (5 фолдов внутри tune), для test — модель, обученная на всём tune.
    def oracle_cut(tau):
        hit = R >= tau - 1e-12
        return np.where(hit.any(1), hit.argmax(1) + 1, t).astype(float)

    make_gbr = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                 subsample=0.8, random_state=seed)
    make_lin = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    # три самых важных признака (по важности бустинга, обученного только на tune при tau=0.9)
    y09 = oracle_cut(0.9)
    gbr09 = make_gbr().fit(feats[tune], y09[tune])
    top3 = np.argsort(-gbr09.feature_importances_)[:3]

    taus = (0.7, 0.8, 0.9, 1.0)

    def add_predictor(name, make, cols):
        preds = {}
        for tau in taus:
            y = oracle_cut(tau)
            x = feats[:, cols]
            pred = np.zeros(n_q)
            for tr, va in KFold(5, shuffle=True, random_state=seed).split(tune):
                pred[tune[va]] = make().fit(x[tune[tr]], y[tune[tr]]).predict(x[tune[va]])
            pred[test] = make().fit(x[tune], y[tune]).predict(x[test])
            preds[tau] = pred
            for delta in np.arange(-3, 15.5, 0.5):
                ms = np.clip(np.round(pred + delta), 1, t).astype(int) - 1
                configs[name].append((f"tau={tau},delta={delta:+.1f}", ms))
        return preds

    all_cols = np.arange(feats.shape[1])
    add_predictor("learned", make_gbr, all_cols)
    lin_preds = add_predictor("linear", make_lin, all_cols)
    add_predictor("lin_top3", make_lin, top3)

    # lin+bud: cut от линейной модели, но не дальше точки, где сумма длин списков достигает бюджета B
    b_sub = b_grid[np.linspace(0, len(b_grid) - 1, 40).astype(int)]
    for tau in taus:
        for delta in np.arange(-3, 12.5, 0.5):
            ms_lin = np.clip(np.round(lin_preds[tau] + delta), 1, t).astype(int) - 1
            for b in b_sub:
                configs["lin+bud"].append((f"tau={tau},d={delta:+.1f},B={int(b)}",
                                           np.minimum(ms_lin, budget_stop[b])))

    fix_rec_q, fix_cost_q = R[test], C[test]
    fix_rec, fix_cost = fix_rec_q.mean(0), fix_cost_q.mean(0)
    fix_p95 = np.percentile(fix_cost_q, 95, axis=0)

    rows = []
    for name, lst in configs.items():
        for params, st in lst:
            for part, sel in (("tune", tune), ("test", test)):
                r, c = evaluate(st[sel], R[sel], C[sel])
                rows.append({"seed": seed, "method": name, "params": params, "part": part, "recall": r.mean(),
                             "cost": c.mean(), "cost_p95": np.percentile(c, 95),
                             "cut_mean": st[sel].mean() + 1})
    by = {(r["method"], r["params"], r["part"]): r for r in rows}

    if verbose:
        print(f"queries: tune {len(tune)}, test {len(test)}")
        print(f"{'target':>6} | {'метод':<9} {'конфиг':<24} {'recall':>6} {'cost':>7} {'p95':>7} {'cut':>5} "
              f"{'ratio':>6}  95% CI")
    results = {}
    for target in args.targets:
        for name, lst in configs.items():
            part = "test" if name == "oracle" else "tune"  # оракулу разрешено смотреть на test: это потолок
            ok = [(p, st) for p, st in lst if by[(name, p, part)]["recall"] >= target]
            if not ok:
                results[(target, name)] = None
                if verbose:
                    print(f"{target:>6.2f} | {name:<9} недостижимо")
                continue
            p, st = min(ok, key=lambda x: (by[(name, x[0], part)][args.select_by], by[(name, x[0], part)]["cost"]))
            res = by[(name, p, "test")]
            base = fixed_interp(fix_rec, fix_cost, res["recall"])
            base_p95 = fixed_interp(fix_rec, fix_p95, res["recall"])
            ratio = res["cost"] / base if np.isfinite(base) and base > 0 else np.nan
            p95r = res["cost_p95"] / base_p95 if np.isfinite(base_p95) and base_p95 > 0 else np.nan
            rq, cq = evaluate(st[test], R[test], C[test])
            lo, hi = (bootstrap_ratio(rq, cq, fix_rec_q, fix_cost_q, args.n_boot, rng)
                      if name != "fixed" else (np.nan, np.nan))
            results[(target, name)] = {"recall": res["recall"], "cost": res["cost"], "p95": res["cost_p95"],
                                       "ratio": ratio, "lo": lo, "hi": hi, "p95r": p95r, "params": p}
            if verbose:
                ci = f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else ""
                print(f"{target:>6.2f} | {name:<9} {p:<24} {res['recall']:>6.3f} {res['cost']:>7.0f} "
                      f"{res['cost_p95']:>7.0f} {res['cut_mean']:>5.1f} {ratio:>6.2f}  {ci}")
        if verbose:
            print()

    if importance:
        print(f"Важность признаков для предсказания cut (tau=0.9, seed {seed}).")
        print("perm = насколько растёт средняя ошибка предсказания cut на test, если перемешать признак;")
        print("gbr = встроенная важность бустинга (доля); ridge = коэффициент линейной модели на стандартизованных признаках\n")
        perm_imp = permutation_importance(gbr09, feats[test], y09[test], scoring="neg_mean_absolute_error",
                                          n_repeats=10, random_state=seed)
        lin09 = make_lin().fit(feats[tune], y09[tune])
        coefs = lin09[-1].coef_
        print(f"{'признак':<16} {'perm':>14} {'gbr':>6} {'ridge':>7}")
        for j in np.argsort(-perm_imp.importances_mean)[:10]:
            print(f"{FEATURE_NAMES[j]:<16} {perm_imp.importances_mean[j]:>7.3f} ±{perm_imp.importances_std[j]:<5.3f} "
                  f"{gbr09.feature_importances_[j]:>6.3f} {coefs[j]:>+7.2f}")
        mae = lambda pr: np.abs(pr - y09[test]).mean()
        print(f"\nсредняя ошибка предсказания cut на test: бустинг {mae(gbr09.predict(feats[test])):.2f}, "
              f"линейная {mae(lin09.predict(feats[test])):.2f}, "
              f"константа (медиана tune) {mae(np.median(y09[tune])):.2f}")
        print("lin_top3 использует признаки:", ", ".join(FEATURE_NAMES[j] for j in top3))
    return results, rows


if __name__ == "__main__":
    main()
