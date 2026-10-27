"""Дешёвая проверка перед «обучением под индекс»: связана ли нужная запросу глубина обхода с тем, насколько
сконцентрированы веса документов, которые попадают в его точный top-10.

Глубина — оракульный N (сколько постингов в порядке убывания вклада нужно, чтобы найти долю tau точного top-10),
как в eval_depth.py. Концентрация документа — доля L1-массы в его m самых тяжёлых координатах (m = 5, 10, 20)
и число ненулевых; для запроса берётся среднее по его top-10.
Выводит: корреляцию Спирмена каждого признака с log N и прирост качества предсказания log N, если к признакам
запроса добавить концентрацию документов (5 фолдов, R^2). Нет связи и нет прироста — идею можно закрывать.
Связь — ещё не причина: это только повод проверять регуляризатор.

  python corr_doc_conc.py --docs docs.npz --queries queries.npz --lam 2000
"""
import argparse

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_cut_v2 import FEATURE_NAMES, coord_stats, query_features
from eval_depth import build_trace, oracle_min, weighted_lists


def row_mass(m, t):
    out = np.zeros(m.shape[0])
    for i in range(m.shape[0]):
        v = np.sort(m.data[m.indptr[i]:m.indptr[i + 1]])[::-1]
        out[i] = v[:t].sum() / v.sum() if v.size else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--lam", type=int, default=2000)
    ap.add_argument("--max-terms", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    k = args.k

    docs = sp.load_npz(args.docs).tocsr()
    docs.sort_indices()
    queries = sp.load_npz(args.queries).tocsr()
    queries.sort_indices()
    csc = docs.tocsc()
    csc.sort_indices()
    df, maxw, thr = coord_stats(docs, args.lam)

    doc_feat = {f"mass{m}_d": row_mass(docs, m) for m in (5, 10, 20)}
    doc_feat["nnz_d"] = np.diff(docs.indptr).astype(float)
    doc_feat["maxshare_d"] = np.array([docs.data[docs.indptr[i]:docs.indptr[i + 1]].max() /
                                       docs.data[docs.indptr[i]:docs.indptr[i + 1]].sum()
                                       if docs.indptr[i + 1] > docs.indptr[i] else 0.0 for i in range(docs.shape[0])])

    items = []
    for qi in range(queries.shape[0]):
        s, e = queries.indptr[qi], queries.indptr[qi + 1]
        idx, val = queries.indices[s:e], queries.data[s:e].astype(np.float64)
        o = np.argsort(-val)
        items.append((idx[o], val[o]))
    lists = weighted_lists(csc, np.unique(np.concatenate([i[:args.max_terms] for i, _ in items])), args.lam)

    logN = {0.9: [], 1.0: []}
    qf, dfeat = [], []
    for idx, val in items:
        t = build_trace(csc, lists, idx, val, k, args.max_terms)
        for tau in logN:
            logN[tau].append(np.log(oracle_min(t.fp_top, tau, k, t.T)))
        scores = np.asarray(csc[:, idx] @ val).ravel()
        top = np.argpartition(-scores, k)[:k]
        dfeat.append([doc_feat[name][top].mean() for name in doc_feat])
        qf.append(query_features(idx[:args.max_terms], val[:args.max_terms], df, maxw, thr, docs.shape[0]))
    qf, dfeat = np.array(qf), np.array(dfeat)
    names = list(doc_feat)

    print(f"docs={docs.shape[0]} queries={queries.shape[0]} lam={args.lam}")
    for tau, y in logN.items():
        y = np.array(y)
        print(f"\n=== глубина для полноты запроса >= {tau}: медиана N = {np.exp(np.median(y)):.0f} постингов ===")
        print(f"{'признак (среднее по top-10)':<28} {'Спирмен rho':>11} {'p':>9}")
        for j, name in enumerate(names):
            rho, p = spearmanr(dfeat[:, j], y)
            print(f"{name:<28} {rho:>11.3f} {p:>9.1e}")
        print("для сравнения — признаки самого запроса:")
        for name in ("nnz", "entropy", "mass_top5", "logdf_t1"):
            j = FEATURE_NAMES.index(name)
            rho, p = spearmanr(qf[:, j], y)
            print(f"  {name:<26} {rho:>11.3f} {p:>9.1e}")
        cv = KFold(5, shuffle=True, random_state=0)
        mk = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        r2 = lambda X: 1 - np.mean((cross_val_predict(mk(), X, y, cv=cv) - y) ** 2) / np.var(y)
        print(f"R^2 предсказания log N (5 фолдов): признаки запроса {r2(qf):.3f}; "
              f"+ концентрация документов {r2(np.hstack([qf, dfeat])):.3f}; только документы {r2(dfeat):.3f}")
    print("\nКонцентрация документов top-10 до поиска неизвестна: это диагностика того, стоит ли учить модель")
    print("концентрировать веса документов, а не признак для предсказания во время поиска.")


if __name__ == "__main__":
    main()
