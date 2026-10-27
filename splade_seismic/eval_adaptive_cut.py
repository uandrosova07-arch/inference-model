"""Оффлайн-оценка адаптивного cut (по доле массы gamma) против фиксированного cut и против оракула.

Это ПРОКСИ-симулятор, а не SEISMIC: нет блоков, summary и heap_factor. Моделируется только эффект cut:
  * индекс: для каждой координаты остаются top-lam документов по весу (статическая обрезка);
  * запрос: берутся его top-cut координат, кандидаты = объединение их урезанных списков;
  * кандидаты оцениваются точно ПОЛНЫМ запросом (как через forward index в SEISMIC);
  * полнота = доля точного top-k, попавшая в кандидаты;
  * стоимость = число кандидатов (число точных скалярных произведений).
Оракул для каждого запроса выбирает наименьший cut, дающий полноту не ниже порога (верхняя граница выигрыша).

Запуск:
  python eval_adaptive_cut.py --synthetic
  python eval_adaptive_cut.py --docs docs.npz --queries queries.npz   (scipy.sparse.save_npz, CSR, N x V и Q x V)
Для реальных данных берите подвыборку документов (до ~1 млн): точный top-k считается перебором.
"""
import argparse
import json

import numpy as np
import scipy.sparse as sp

from adaptive_cut import gamma_cut


def make_synthetic(seed: int, n_docs: int, vocab: int, n_queries: int):
    """Синтетика с тяжёлыми хвостами весов; запросы двух типов: концентрированные и плоские.
    Служит только для проверки кода, выводов о реальных данных из неё делать нельзя."""
    rng = np.random.default_rng(seed)
    pop = 1.0 / (np.arange(vocab) + 1.0) ** 0.9
    pop /= pop.sum()

    rows, cols, vals = [], [], []
    doc_terms = []
    for i in range(n_docs):
        terms = rng.choice(vocab, size=int(rng.integers(40, 90)), replace=False, p=pop)
        doc_terms.append(terms)
        rows.append(np.full(len(terms), i))
        cols.append(terms)
        vals.append(rng.lognormal(0.0, 1.0, size=len(terms)))
    docs = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n_docs, vocab))

    rows, cols, vals = [], [], []
    for i in range(n_queries):
        base = doc_terms[rng.integers(n_docs)]
        keep = base[rng.random(len(base)) < 0.6]
        noise = rng.choice(vocab, size=5, replace=False, p=pop)
        terms = np.unique(np.concatenate([keep, noise]))
        sigma = 1.6 if i % 2 == 0 else 0.2  # концентрированный / плоский запрос
        rows.append(np.full(len(terms), i))
        cols.append(terms)
        vals.append(rng.lognormal(0.0, sigma, size=len(terms)))
    queries = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(n_queries, vocab))
    return docs, queries


def build_postings(docs: sp.csr_matrix, lam: int):
    """Для каждой координаты: id top-lam документов по весу."""
    csc = docs.tocsc()
    postings = []
    for j in range(csc.shape[1]):
        s, e = csc.indptr[j], csc.indptr[j + 1]
        ids, w = csc.indices[s:e], csc.data[s:e]
        if len(ids) > lam:
            ids = ids[np.argpartition(-w, lam)[:lam]]
        postings.append(ids)
    return postings


def per_query_curves(docs, queries, postings, k: int, max_cut: int):
    """recall[q, m-1], cost[q, m-1] для m = 1..max_cut и отсортированные веса каждого запроса."""
    n_docs = docs.shape[0]
    n_q = queries.shape[0]
    recall = np.zeros((n_q, max_cut))
    cost = np.zeros((n_q, max_cut))
    qvals = []
    for qi in range(n_q):
        s, e = queries.indptr[qi], queries.indptr[qi + 1]
        idx, val = queries.indices[s:e], queries.data[s:e]
        order = np.argsort(-val)
        idx, val = idx[order], val[order]
        qvals.append(val)

        scores = docs @ sp.csr_matrix((val, (np.zeros(len(idx), dtype=int), idx)), shape=(1, docs.shape[1])).T
        scores = scores.toarray().ravel()
        topk = np.argpartition(-scores, k)[:k]

        mask = np.zeros(n_docs, dtype=bool)
        count = 0
        for m in range(max_cut):
            if m < len(idx):  # дальше последней координаты запроса кандидаты не меняются
                ids = postings[idx[m]]
                count += int((~mask[ids]).sum())
                mask[ids] = True
            recall[qi, m] = mask[topk].mean()
            cost[qi, m] = count
    return recall, cost, qvals


def point(recall, cost, m_per_query):
    rows = np.arange(len(m_per_query))
    r, c = recall[rows, m_per_query - 1], cost[rows, m_per_query - 1]
    return {"recall": float(r.mean()), "cost": float(c.mean()), "cost_p95": float(np.percentile(c, 95)),
            "cut_mean": float(m_per_query.mean())}


def cheapest(points, target):
    ok = [p for p in points if p["recall"] >= target]
    return min(ok, key=lambda p: p["cost"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--docs")
    ap.add_argument("--queries")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lam", type=int, default=200, help="длина списка после статической обрезки")
    ap.add_argument("--max-cut", type=int, default=30)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.97])
    ap.add_argument("--n-docs", type=int, default=20000)
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    if args.synthetic:
        docs, queries = make_synthetic(args.seed, args.n_docs, args.vocab, args.n_queries)
    else:
        docs, queries = sp.load_npz(args.docs).tocsr(), sp.load_npz(args.queries).tocsr()

    postings = build_postings(docs, args.lam)
    recall, cost, qvals = per_query_curves(docs, queries, postings, args.k, args.max_cut)
    n_q = len(qvals)

    fixed = []
    for m in range(1, args.max_cut + 1):
        p = point(recall, cost, np.full(n_q, m))
        p["param"] = f"cut={m}"
        fixed.append(p)

    adaptive = []
    for g in np.linspace(0.05, 1.0, 96):
        ms = np.array([gamma_cut(v, g, 1, args.max_cut) for v in qvals])
        p = point(recall, cost, ms)
        p["param"] = f"gamma={g:.2f}"
        adaptive.append(p)

    oracle = []
    for thr in np.linspace(0.5, 1.0, 51):
        ms = np.array([(int(np.argmax(recall[q] >= thr)) + 1) if (recall[q] >= thr).any() else args.max_cut
                       for q in range(n_q)])
        p = point(recall, cost, ms)
        p["param"] = f"per-query recall>={thr:.2f}"
        oracle.append(p)

    print(f"docs={docs.shape[0]} queries={n_q} k={args.k} lam={args.lam} max_cut={args.max_cut}")
    print("стоимость = среднее число кандидатов на запрос (прокси числа точных скалярных произведений)\n")
    print(f"{'target':>7} | {'метод':<10} {'конфиг':<28} {'recall':>7} {'cost':>9} {'cost p95':>9} {'cut':>6}")
    result = {}
    for t in args.targets:
        base = cheapest(fixed, t)
        for name, pts in (("fixed", fixed), ("adaptive", adaptive), ("oracle", oracle)):
            p = cheapest(pts, t)
            if p is None:
                print(f"{t:>7.2f} | {name:<10} недостижимо при max_cut={args.max_cut}")
                continue
            gain = "" if base is None or name == "fixed" else f"  ({p['cost'] / base['cost']:.2f}x от fixed)"
            print(f"{t:>7.2f} | {name:<10} {p['param']:<28} {p['recall']:>7.3f} {p['cost']:>9.0f} "
                  f"{p['cost_p95']:>9.0f} {p['cut_mean']:>6.1f}{gain}")
            result[f"{name}@{t}"] = p
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
