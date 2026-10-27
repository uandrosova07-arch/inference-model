"""Сквозное время, часть 2: поиск в SEISMIC (включая точный досчёт кандидатов по forward index) и его качество.

Для каждой настройки (query_cut, heap_factor): время одного запроса (минимум из --repeats, один поток, по --n-time
запросам), полнота относительно точного top-10 той же модели и MRR@10 по разметке dev — на всех запросах.
Отдельно — MRR@10 точного поиска (верхняя граница качества модели).

  python e2e_search.py --enc ENC --sub SUB --zip msmarco.zip --n-postings 2000 --out e2e_search.json
"""
import argparse
import json
import os
import time
import zipfile

import numpy as np
import scipy.sparse as sp

CONFIGS = [(1, 1.0), (2, 0.95), (3, 0.95), (4, 0.9), (5, 0.9), (5, 0.85), (6, 1.0), (8, 1.0), (10, 1.0)]


def read_jsonl(path, key):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[key] for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n-postings", type=int, default=2000)
    ap.add_argument("--n-time", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import seismic

    docs = sp.load_npz(os.path.join(args.enc, "docs.npz")).tocsr()
    docs.sort_indices()
    queries = sp.load_npz(os.path.join(args.enc, "queries.npz")).tocsr()
    queries.sort_indices()
    q_ids = read_jsonl(os.path.join(args.sub, "queries.jsonl"), "_id")
    doc_ids = read_jsonl(os.path.join(args.sub, "corpus.jsonl"), "_id")
    qrels = {}
    with zipfile.ZipFile(args.zip) as z:
        with z.open(next(n for n in z.namelist() if n.endswith("qrels/dev.tsv"))) as f:
            next(f)
            for line in f:
                q, d, s = line.decode("utf-8").rstrip("\n").split("\t")
                if int(s) > 0:
                    qrels.setdefault(q, set()).add(d)

    def mrr_of(ranked_lists):
        tot, n = 0.0, 0
        for qi, ranked in enumerate(ranked_lists):
            rel = qrels.get(q_ids[qi])
            if rel:
                first = next((r for r, j in enumerate(ranked[:10]) if doc_ids[j] in rel), None)
                tot += 0.0 if first is None else 1.0 / (first + 1)
                n += 1
        return round(tot / max(n, 1), 4)

    docs_t = docs.T.tocsr()
    exact = []
    for st in range(0, queries.shape[0], 100):
        block = (queries[st:st + 100] @ docs_t).toarray()
        for row in block:
            t = np.argpartition(-row, args.k)[:args.k]
            exact.append(t[np.argsort(-row[t])])
    results = [{"config": "exact", "recall@10": 1.0, "MRR@10": mrr_of(exact)}]

    stype = seismic.get_seismic_string()
    ds = seismic.SeismicDataset(dim=docs.shape[1])
    ds.add_documents_by_id([str(i) for i in range(docs.shape[0])], docs.indices.astype(np.int64),
                           docs.data.astype(np.float32), docs.indptr.astype(np.int64))
    index = seismic.SeismicIndex.build_from_dataset(ds, n_postings=args.n_postings, num_threads=args.build_threads)
    comps, vals = [], []
    for qi in range(queries.shape[0]):
        s, e = queries.indptr[qi], queries.indptr[qi + 1]
        comps.append(np.array([str(i) for i in queries.indices[s:e]], dtype=stype))
        vals.append(queries.data[s:e].astype(np.float32))

    # точный перебор в том же движке — ориентир по времени
    bf = []
    for qi in range(min(50, queries.shape[0])):
        t0 = time.perf_counter_ns()
        ds.search(str(qi), comps[qi], vals[qi], args.k)
        bf.append((time.perf_counter_ns() - t0) / 1000.0)
    results[0].update({"median_us": round(float(np.median(bf)), 1), "p95_us": round(float(np.percentile(bf, 95)), 1)})

    n_time = min(args.n_time, queries.shape[0])
    for qi in range(n_time):
        index.search(str(qi), comps[qi], vals[qi], args.k, 5, 0.9)
    for c, h in CONFIGS:
        lat, ranked, rec = [], [], []
        for qi in range(queries.shape[0]):
            best = np.inf
            for _ in range(args.repeats if qi < n_time else 1):
                t0 = time.perf_counter_ns()
                res = index.search(str(qi), comps[qi], vals[qi], args.k, c, h)
                best = min(best, time.perf_counter_ns() - t0)
            if qi < n_time:
                lat.append(best / 1000.0)
            ids = [int(r[2]) for r in res]
            ranked.append(ids)
            rec.append(len(set(ids) & set(exact[qi].tolist())) / args.k)
        r = {"config": f"cut={c},hf={h}", "recall@10": round(float(np.mean(rec)), 4), "MRR@10": mrr_of(ranked),
             "median_us": round(float(np.median(lat)), 1), "p95_us": round(float(np.percentile(lat, 95)), 1)}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
