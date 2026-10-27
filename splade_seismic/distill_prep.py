"""Данные для дистилляции маленького кодировщика запроса под ГОТОВЫЙ индекс документов.

Берёт --n train-запросов MS MARCO (без запросов из dev-подвыборки), кодирует их учителем (тот же SPLADE, что строил
индекс) и для каждого находит --topk лучших документов индекса точным поиском на GPU.
Сохраняет: texts.json, teacher_q.npz (векторы учителя), cand.npy (n x topk номеров документов).

  python distill_prep.py --zip msmarco.zip --enc ENC --sub SUB --out-dir distill_data --n 100000
"""
import argparse
import json
import os
import zipfile

import numpy as np
import scipy.sparse as sp
import torch

from qeval import read_jsonl
from splade import Splade, encode_to_csr, tokenizer_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="naver/splade-cocondenser-ensembledistil")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--enc", required=True, help="папка с docs.npz индекса")
    ap.add_argument("--sub", required=True, help="папка с queries.jsonl dev-подвыборки (их исключаем)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--max-len-q", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dev_ids = set(read_jsonl(os.path.join(args.sub, "queries.jsonl"), "_id"))
    with zipfile.ZipFile(args.zip) as z:
        with z.open(next(n for n in z.namelist() if n.endswith("qrels/train.tsv"))) as f:
            next(f)
            train_ids = sorted({line.decode("utf-8").split("\t")[0] for line in f} - dev_ids)
        with z.open(next(n for n in z.namelist() if n.endswith("queries.jsonl"))) as f:
            text_of = {}
            for line in f:
                q = json.loads(line)
                text_of[q["_id"]] = q["text"]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(train_ids), size=min(args.n, len(train_ids)), replace=False)
    texts = [text_of[train_ids[i]] for i in pick]
    print(f"train-запросов выбрано: {len(texts)}", flush=True)

    tok = tokenizer_for(args.teacher)
    teacher = Splade(args.teacher).to(args.device).eval()
    tq = encode_to_csr(teacher, tok, texts, args.max_len_q, 256, args.device, fp16=True, log_every=100)
    del teacher
    print(f"учитель: в среднем {tq.nnz / tq.shape[0]:.1f} ненулевых на запрос", flush=True)

    docs = sp.load_npz(os.path.join(args.enc, "docs.npz")).tocsr()
    d = torch.sparse_csr_tensor(torch.as_tensor(docs.indptr, dtype=torch.int64),
                                torch.as_tensor(docs.indices, dtype=torch.int64),
                                torch.as_tensor(docs.data, dtype=torch.float32),
                                size=docs.shape).to(args.device)
    cand = np.zeros((len(texts), args.topk), dtype=np.int32)
    bs = 256
    for st in range(0, len(texts), bs):
        q = torch.as_tensor(tq[st:st + bs].toarray(), device=args.device)
        scores = torch.sparse.mm(d, q.T)  # docs x B
        cand[st:st + bs] = torch.topk(scores, args.topk, dim=0).indices.T.cpu().numpy()
        if (st // bs) % 50 == 0:
            print(f"  кандидаты: {st + len(q)}/{len(texts)}", flush=True)
    with open(os.path.join(args.out_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    sp.save_npz(os.path.join(args.out_dir, "teacher_q.npz"), tq)
    np.save(os.path.join(args.out_dir, "cand.npy"), cand)
    print("готово", flush=True)


if __name__ == "__main__":
    main()
