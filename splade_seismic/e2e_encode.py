"""Сквозное время, часть 1: сколько стоит превратить текст запроса в разреженный вектор.

Варианты кодировщика (время на ОДИН запрос, batch = 1, медиана и p95 по --n-time запросам после прогрева):
  cpu-fp32-1t / cpu-fp32-4t — полный SPLADE на CPU, 1 и 4 потока;
  cpu-int8-1t               — то же с динамическим int8-квантованием линейных слоёв;
  cpu-L4-1t / cpu-L2-1t     — только первые 4 / 2 слоя BERT: НИЖНЯЯ ГРАНИЦА времени маленького кодировщика
                              запроса (без дистилляции качество не имеет смысла, его не считаем);
  gpu-fp32 / gpu-fp16       — на GPU (если есть);
  inference-free            — токенизация + веса слов из таблицы: нижняя граница для inference-free SPLADE.
Для полных вариантов (fp32, int8, fp16) считается и качество на всех запросах: MRR@10 по разметке dev при точном
поиске и совпадение top-10 с исходными векторами запросов (queries.npz).

  python e2e_encode.py --model naver/splade-cocondenser-ensembledistil --enc ENC --sub SUB --zip msmarco.zip --out e2e_encode.json
"""
import argparse
import copy
import warnings
import json
import os
import time
import zipfile

import numpy as np
import scipy.sparse as sp
import torch

from splade import Splade, tokenizer_for


def read_jsonl(path, key):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[key] for line in f]


def time_fn(fn, texts, warmup=20):
    for t in texts[:warmup]:
        fn(t)
    out = []
    for t in texts:
        t0 = time.perf_counter_ns()
        fn(t)
        out.append((time.perf_counter_ns() - t0) / 1000.0)
    return float(np.median(out)), float(np.percentile(out, 95))


def truncated(model, n_layers):
    m = copy.deepcopy(model)
    base = getattr(m.mlm, m.mlm.base_model_prefix)
    base.encoder.layer = torch.nn.ModuleList(list(base.encoder.layer)[:n_layers])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--enc", required=True, help="папка с docs.npz и queries.npz (исходные векторы)")
    ap.add_argument("--sub", required=True, help="папка с corpus.jsonl и queries.jsonl")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n-time", type=int, default=300)
    ap.add_argument("--max-len-q", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")  # предупреждения torch об устаревшем API квантования

    q_texts = read_jsonl(os.path.join(args.sub, "queries.jsonl"), "text")
    q_ids = read_jsonl(os.path.join(args.sub, "queries.jsonl"), "_id")
    doc_ids = read_jsonl(os.path.join(args.sub, "corpus.jsonl"), "_id")
    docs_t = sp.load_npz(os.path.join(args.enc, "docs.npz")).tocsr().T.tocsr()
    q_ref = sp.load_npz(os.path.join(args.enc, "queries.npz")).tocsr()
    qrels = {}
    with zipfile.ZipFile(args.zip) as z:
        with z.open(next(n for n in z.namelist() if n.endswith("qrels/dev.tsv"))) as f:
            next(f)
            for line in f:
                q, d, s = line.decode("utf-8").rstrip("\n").split("\t")
                if int(s) > 0:
                    qrels.setdefault(q, set()).add(d)

    def top_k(qmat):
        res = []
        for st in range(0, qmat.shape[0], 100):
            block = (qmat[st:st + 100] @ docs_t).toarray()
            for row in block:
                t = np.argpartition(-row, args.k)[:args.k]
                res.append(t[np.argsort(-row[t])])
        return res

    ref_top = top_k(q_ref)

    def quality(qmat):
        tops = top_k(qmat)
        mrr, n, overlap = 0.0, 0, []
        for qi, t in enumerate(tops):
            overlap.append(len(set(t.tolist()) & set(ref_top[qi].tolist())) / args.k)
            rel = qrels.get(q_ids[qi])
            if rel:
                first = next((r for r, j in enumerate(t) if doc_ids[j] in rel), None)
                mrr += 0.0 if first is None else 1.0 / (first + 1)
                n += 1
        return {"MRR@10": round(mrr / max(n, 1), 4), "overlap@10_vs_ref": round(float(np.mean(overlap)), 4)}

    tok = tokenizer_for(args.model)
    base_cpu = Splade(args.model).eval()
    timing_texts = q_texts[:args.n_time]
    results = []

    def encoder(model, device, amp=False):
        @torch.no_grad()
        def enc(text):
            b = tok([text], truncation=True, max_length=args.max_len_q, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                w = model(b["input_ids"], b["attention_mask"])
            w = w.float()
            if device != "cpu":
                torch.cuda.synchronize()
            nz = torch.nonzero(w[0]).squeeze(1)
            return nz.cpu().numpy(), w[0, nz].cpu().numpy()
        return enc

    def encode_all(enc):
        rows, cols, vals = [], [], []
        for i, t in enumerate(q_texts):
            c, v = enc(t)
            rows.append(np.full(len(c), i)); cols.append(c); vals.append(v)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(q_texts), q_ref.shape[1]))

    def record(name, enc, with_quality):
        med, p95 = time_fn(enc, timing_texts)
        r = {"variant": name, "median_us": round(med, 1), "p95_us": round(p95, 1)}
        if with_quality:
            r.update(quality(encode_all(enc)))
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)

    torch.set_num_threads(1)
    record("cpu-fp32-1t", encoder(base_cpu, "cpu"), True)
    torch.set_num_threads(4)
    record("cpu-fp32-4t", encoder(base_cpu, "cpu"), False)
    torch.set_num_threads(1)
    q8 = torch.quantization.quantize_dynamic(copy.deepcopy(base_cpu), {torch.nn.Linear}, dtype=torch.qint8)
    record("cpu-int8-1t", encoder(q8, "cpu"), True)
    for n_layers in (4, 2):
        record(f"cpu-L{n_layers}-1t", encoder(truncated(base_cpu, n_layers), "cpu"), False)

    if torch.cuda.is_available():
        gpu = copy.deepcopy(base_cpu).to("cuda")
        record("gpu-fp32", encoder(gpu, "cuda"), False)
        record("gpu-fp16", encoder(gpu, "cuda", amp=True), True)

    table = np.random.default_rng(0).random(q_ref.shape[1]).astype(np.float32)

    def inference_free(text):
        ids = tok(text, truncation=True, max_length=args.max_len_q)["input_ids"]
        u = np.unique(np.asarray(ids, dtype=np.int64))
        return u, table[u]

    record("inference-free", inference_free, False)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
