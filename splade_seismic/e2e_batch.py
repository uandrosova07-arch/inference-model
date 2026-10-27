"""Кодирование запросов SPLADE пачками: время на один запрос при пропускной способности и задержка пачки.

Запросы берутся в случайном порядке (как пришли бы в систему), пачка дополняется паддингом до самого длинного.
В замер входит всё: токенизация, перенос на устройство, модель, извлечение ненулевых на CPU.
Варианты: GPU fp32, GPU fp16 через model.half() (настоящие half-веса), GPU fp16 через autocast (как в
e2e_encode.py — для сравнения), CPU fp32 в 4 потока. Для fp16 проверяется косинус с исходными векторами.

  python e2e_batch.py --model naver/splade-cocondenser-ensembledistil --enc ENC --sub SUB --out e2e_batch.json
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

from splade import Splade, tokenizer_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--enc", required=True, help="папка с queries.npz (исходные векторы запросов)")
    ap.add_argument("--sub", required=True, help="папка с queries.jsonl")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    ap.add_argument("--n", type=int, default=2000, help="сколько запросов прогонять (для batch=1 — не больше 500)")
    ap.add_argument("--max-len-q", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.sub, "queries.jsonl"), encoding="utf-8") as f:
        all_texts = [json.loads(line)["text"] for line in f]
    order = np.random.default_rng(0).permutation(len(all_texts))[:args.n]
    texts = [all_texts[i] for i in order]
    q_ref = sp.load_npz(os.path.join(args.enc, "queries.npz")).tocsr()[order]
    tok = tokenizer_for(args.model)
    base = Splade(args.model).eval()
    results = []

    def run(name, model, device, bs, n_texts, amp=False):
        sub = texts[:n_texts]

        @torch.no_grad()
        def one(chunk):
            t0 = time.perf_counter_ns()
            b = tok(chunk, padding=True, truncation=True, max_length=args.max_len_q, return_tensors="pt")
            t1 = time.perf_counter_ns()
            b = b.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                w = model(b["input_ids"], b["attention_mask"])
            w = w.float()
            r, c = torch.nonzero(w, as_tuple=True)
            v = w[r, c]
            out = r.cpu().numpy(), c.cpu().numpy(), v.cpu().numpy()  # .cpu() ждёт окончания счёта на GPU
            return out, t1 - t0, time.perf_counter_ns() - t0

        for i in range(0, min(len(sub), max(bs, 20 * bs if bs < 32 else 3 * bs)), bs):  # прогрев
            one(sub[i:i + bs])
        lat, tok_ns, rows, cols, vals = [], 0, [], [], []
        t_start = time.perf_counter_ns()
        for i in range(0, len(sub), bs):
            (r, c, v), tk, total = one(sub[i:i + bs])
            lat.append(total / 1e6)
            tok_ns += tk
            rows.append(r + i); cols.append(c); vals.append(v)
        wall = time.perf_counter_ns() - t_start
        res = {"variant": name, "batch": bs, "n": len(sub),
               "us_per_query": round(wall / 1000.0 / len(sub), 1),
               "batch_median_ms": round(float(np.median(lat)), 2),
               "batch_p95_ms": round(float(np.percentile(lat, 95)), 2),
               "tokenize_share": round(tok_ns / wall, 3)}
        if "fp16" in name:
            m = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(sub), q_ref.shape[1]))
            ref = q_ref[:len(sub)]
            num = np.asarray(m.multiply(ref).sum(axis=1)).ravel()
            den = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel() * np.asarray(ref.multiply(ref).sum(axis=1)).ravel())
            res["cos_vs_ref_min"] = round(float((num / np.maximum(den, 1e-12)).min()), 5)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        results.append(res)

    if torch.cuda.is_available():
        g32 = copy.deepcopy(base).to("cuda")
        for bs in args.batch_sizes:
            run("gpu-fp32", g32, "cuda", bs, min(args.n, 500) if bs == 1 else args.n)
        g16 = copy.deepcopy(base).half().to("cuda")
        for bs in args.batch_sizes:
            run("gpu-fp16-half", g16, "cuda", bs, min(args.n, 500) if bs == 1 else args.n)
        del g16
        for bs in (1, 128):
            run("gpu-fp16-autocast", g32, "cuda", bs, min(args.n, 500) if bs == 1 else args.n, amp=True)
        del g32
        torch.cuda.empty_cache()
    torch.set_num_threads(4)
    for bs in (1, 8, 32):
        run("cpu-fp32-4t", base, "cpu", bs, 256)

    print("\n=== Время на запрос при кодировании пачками (мкс/запрос — пропускная способность; "
          "задержка — время одной пачки) ===")
    print(f"{'вариант':<20} {'пачка':>6} {'мкс/запрос':>11} {'пачка, мед. мс':>15} {'пачка, p95 мс':>14} "
          f"{'токенизация':>12} {'cos min':>8}")
    for r in results:
        cos = f"{r['cos_vs_ref_min']:>8.4f}" if "cos_vs_ref_min" in r else f"{'—':>8}"
        print(f"{r['variant']:<20} {r['batch']:>6} {r['us_per_query']:>11.0f} {r['batch_median_ms']:>15.2f} "
              f"{r['batch_p95_ms']:>14.2f} {100 * r['tokenize_share']:>11.1f}% {cos}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
