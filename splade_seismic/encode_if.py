"""Inference-free SPLADE (OpenSearch doc-v2-distill): документы кодирует нейросеть, запрос — без нейросети.

Документ: MLM-логиты DistilBERT -> max по токенам -> log(1+ReLU), служебные токены обнуляются (как в карточке модели).
Запрос: токены запроса без служебных, вес каждого токена — его IDF из idf.json модели (без расширений).
Сохраняет docs.npz и queries.npz в том же формате, что encode_corpus.py, и время кодирования одного запроса
(токенизация + IDF + сборка вектора) в формате e2e_encode.json.

  python encode_if.py --sub SUB --out-dir ENC_IF --timing-out if_encode.json --fp16
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

from splade import Splade, encode_to_csr, tokenizer_for

MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_idf(model, tok, vocab_size, idf_file=None):
    if idf_file is None:
        from huggingface_hub import hf_hub_download
        idf_file = hf_hub_download(repo_id=model, filename="idf.json")
    idf = np.zeros(vocab_size, dtype=np.float32)
    for token, w in json.load(open(idf_file, encoding="utf-8")).items():
        idf[tok.convert_tokens_to_ids(token)] = w
    idf[tok.all_special_ids] = 0.0
    return idf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--idf-file", help="локальный idf.json (по умолчанию берётся из репозитория модели)")
    ap.add_argument("--sub", required=True, help="папка с corpus.jsonl и queries.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timing-out", required=True)
    ap.add_argument("--max-len-d", type=int, default=512)
    ap.add_argument("--max-len-q", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-time", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = tokenizer_for(args.model)
    model = Splade(args.model).to(args.device)
    vocab = model.mlm.config.vocab_size
    special = np.array(tok.all_special_ids)

    docs = read_jsonl(os.path.join(args.sub, "corpus.jsonl"))
    doc_texts = [(d.get("title", "") + " " + d["text"]).strip() for d in docs]
    print(f"кодирую {len(doc_texts)} документов на {args.device}", flush=True)
    d_mat = encode_to_csr(model, tok, doc_texts, args.max_len_d, args.batch_size, args.device,
                          fp16=args.fp16, log_every=500)
    d_mat.data[np.isin(d_mat.indices, special)] = 0.0
    d_mat.eliminate_zeros()
    del model

    idf = load_idf(args.model, tok, vocab, args.idf_file)

    def enc_q(text):
        ids = tok(text, truncation=True, max_length=args.max_len_q, add_special_tokens=False)["input_ids"]
        u = np.unique(np.asarray(ids, dtype=np.int64))
        v = idf[u]
        keep = v > 0
        return u[keep], v[keep]

    q_texts = [q["text"] for q in read_jsonl(os.path.join(args.sub, "queries.jsonl"))]
    rows, cols, vals = [], [], []
    for i, t in enumerate(q_texts):
        c, v = enc_q(t)
        rows.append(np.full(len(c), i)); cols.append(c); vals.append(v)
    q_mat = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(q_texts), vocab))
    sp.save_npz(os.path.join(args.out_dir, "docs.npz"), d_mat)
    sp.save_npz(os.path.join(args.out_dir, "queries.npz"), q_mat)
    print(f"docs {d_mat.shape}, среднее число ненулевых {d_mat.nnz / d_mat.shape[0]:.1f}; "
          f"queries {q_mat.shape}, среднее число ненулевых {q_mat.nnz / q_mat.shape[0]:.1f}", flush=True)

    timing_texts = q_texts[:args.n_time]
    for t in timing_texts[:20]:
        enc_q(t)
    lat = []
    for t in timing_texts:
        t0 = time.perf_counter_ns()
        enc_q(t)
        lat.append((time.perf_counter_ns() - t0) / 1000.0)
    r = {"variant": "if-idf-real", "median_us": round(float(np.median(lat)), 1),
         "p95_us": round(float(np.percentile(lat, 95)), 1)}
    print(json.dumps(r, ensure_ascii=False))
    with open(args.timing_out, "w", encoding="utf-8") as f:
        json.dump([r], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
