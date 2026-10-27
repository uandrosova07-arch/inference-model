"""Дистилляция маленького кодировщика запроса SPLADE под готовый индекс документов (документы не трогаем).

Ученик — маленький BERT с MLM-головой на том же словаре (например google/bert_uncased_L-4_H-256_A-4); выход —
как у SPLADE: log(1+ReLU(логиты)), max по токенам.
Потери на пачке из B запросов:
  kl  — KL(учитель || ученик) по распределению скоров на общем пуле документов (кандидаты всех запросов пачки,
        то есть свои кандидаты + чужие как отрицательные); документы — настоящие векторы индекса;
  mse — ||ученик - учитель||^2 / ||учитель||^2 по векторам запросов (держит ученика в «словаре» учителя);
  l1  — по желанию, за длину вектора запроса (короче запрос — быстрее поиск).

  python distill_train.py --student google/bert_uncased_L-4_H-256_A-4 --data distill_data --docs ENC/docs.npz --out stud_mini
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from splade import Splade, tokenizer_for


def coo_rows(m, device):
    m = m.tocoo()
    idx = torch.as_tensor(np.vstack([m.row, m.col]), dtype=torch.int64)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(m.data, dtype=torch.float32), m.shape).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--max-len-q", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    texts = json.load(open(os.path.join(args.data, "texts.json"), encoding="utf-8"))
    tq = sp.load_npz(os.path.join(args.data, "teacher_q.npz")).tocsr()
    cand = np.load(os.path.join(args.data, "cand.npy"))
    docs = sp.load_npz(args.docs).tocsr()
    assert docs.shape[1] == tq.shape[1]

    tok = tokenizer_for(args.student)
    assert len(tok) == tq.shape[1] or tok.vocab_size == tq.shape[1], "словарь ученика должен совпадать со словарём учителя"
    model = Splade(args.student).to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.0, 1 - s / args.steps))

    order, pos = rng.permutation(len(texts)), 0
    t0, logs = time.time(), []
    for step in range(1, args.steps + 1):
        if pos + args.batch_size > len(order):
            order, pos = rng.permutation(len(texts)), 0
        bi = order[pos:pos + args.batch_size]
        pos += args.batch_size
        b = tok([texts[i] for i in bi], padding=True, truncation=True, max_length=args.max_len_q,
                return_tensors="pt").to(args.device)
        s = model(b["input_ids"], b["attention_mask"])                       # B x V
        t = torch.as_tensor(tq[bi].toarray(), device=args.device)            # B x V
        pool = np.unique(cand[bi].ravel())
        dp = coo_rows(docs[pool], args.device)                               # P x V
        st = torch.sparse.mm(dp, t.T).T                                      # B x P
        ss = torch.sparse.mm(dp, s.T).T
        kl = F.kl_div(F.log_softmax(ss / args.temp, dim=1), F.log_softmax(st / args.temp, dim=1),
                      log_target=True, reduction="batchmean")
        mse = ((s - t) ** 2).sum(1).div(t.pow(2).sum(1).clamp_min(1e-6)).mean()
        l1 = s.sum(1).mean()
        loss = args.w_kl * kl + args.w_mse * mse + args.l1 * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            nnz = (s.detach() > 0).sum(1).float().mean().item()
            rec = {"step": step, "kl": round(kl.item(), 4), "mse": round(mse.item(), 4), "nnz_student": round(nnz, 1),
                   "nnz_teacher": round(tq[bi].nnz / len(bi), 1), "min": round((time.time() - t0) / 60, 1)}
            logs.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.mlm.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "log": logs}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
