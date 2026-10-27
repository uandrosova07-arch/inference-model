"""Общие функции для замеров кодировщика запроса: качество (MRR@10, совпадение top-10) и время."""
import json
import os
import time
import zipfile

import numpy as np
import scipy.sparse as sp


def read_jsonl(path, key=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r[key] for r in rows] if key else rows


def load_qrels(zip_path, split="dev"):
    qrels = {}
    with zipfile.ZipFile(zip_path) as z:
        with z.open(next(n for n in z.namelist() if n.endswith(f"qrels/{split}.tsv"))) as f:
            next(f)
            for line in f:
                q, d, s = line.decode("utf-8").rstrip("\n").split("\t")
                if int(s) > 0:
                    qrels.setdefault(q, set()).add(d)
    return qrels


class Evaluator:
    """Точный поиск по docs.npz; качество векторов запросов относительно разметки dev и исходных векторов."""

    def __init__(self, enc_dir, sub_dir, zip_path, k=10):
        self.k = k
        self.docs_t = sp.load_npz(os.path.join(enc_dir, "docs.npz")).tocsr().T.tocsr()
        self.q_ref = sp.load_npz(os.path.join(enc_dir, "queries.npz")).tocsr()
        self.vocab = self.q_ref.shape[1]
        self.q_texts = read_jsonl(os.path.join(sub_dir, "queries.jsonl"), "text")
        self.q_ids = read_jsonl(os.path.join(sub_dir, "queries.jsonl"), "_id")
        self.doc_ids = read_jsonl(os.path.join(sub_dir, "corpus.jsonl"), "_id")
        self.qrels = load_qrels(zip_path, "dev")
        self.ref_top = self.top_k(self.q_ref)

    def top_k(self, qmat):
        res = []
        for st in range(0, qmat.shape[0], 100):
            block = (qmat[st:st + 100] @ self.docs_t).toarray()
            for row in block:
                t = np.argpartition(-row, self.k)[:self.k]
                res.append(t[np.argsort(-row[t])])
        return res

    def quality(self, qmat):
        tops = self.top_k(qmat)
        mrr, n, overlap = 0.0, 0, []
        for qi, t in enumerate(tops):
            overlap.append(len(set(t.tolist()) & set(self.ref_top[qi].tolist())) / self.k)
            rel = self.qrels.get(self.q_ids[qi])
            if rel:
                first = next((r for r, j in enumerate(t) if self.doc_ids[j] in rel), None)
                mrr += 0.0 if first is None else 1.0 / (first + 1)
                n += 1
        return {"MRR@10": round(mrr / max(n, 1), 4), "overlap@10_vs_ref": round(float(np.mean(overlap)), 4),
                "nnz_q": round(qmat.nnz / qmat.shape[0], 1)}


def time_fn(fn, texts, warmup=20):
    for t in texts[:warmup]:
        fn(t)
    out = []
    for t in texts:
        t0 = time.perf_counter_ns()
        fn(t)
        out.append((time.perf_counter_ns() - t0) / 1000.0)
    return round(float(np.median(out)), 1), round(float(np.percentile(out, 95)), 1)


def encode_all(enc, texts, vocab):
    """enc(text) -> (индексы, веса); собирает CSR len(texts) x vocab."""
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        c, v = enc(t)
        rows.append(np.full(len(c), i)); cols.append(np.asarray(c)); vals.append(np.asarray(v, dtype=np.float32))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(texts), vocab))
