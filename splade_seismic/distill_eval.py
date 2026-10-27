"""Оценка учеников-кодировщиков запроса против учителя на ТОМ ЖЕ индексе документов.

Для каждой модели: время одного запроса (CPU 1 поток, CPU 4 потока, GPU fp16 batch=1), MRR@10 при точном поиске,
совпадение top-10 с учителем, длина вектора запроса. Векторы запросов каждой модели сохраняются в
OUT_ROOT/<имя>/queries.npz (рядом — ссылка на docs.npz индекса), чтобы прогнать по ним e2e_search.py.

  python distill_eval.py --models naver/splade-cocondenser-ensembledistil stud_tiny stud_mini --enc ENC --sub SUB \
      --zip msmarco.zip --out-root stud_enc --out distill_eval.json
"""
import argparse
import json
import os

import scipy.sparse as sp
import torch

from qeval import Evaluator, encode_all, time_fn
from splade import Splade, tokenizer_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-time", type=int, default=300)
    ap.add_argument("--max-len-q", type=int, default=64)
    args = ap.parse_args()

    ev = Evaluator(args.enc, args.sub, args.zip)
    timing_texts = ev.q_texts[:args.n_time]
    results = []
    for path in args.models:
        name = os.path.basename(path.rstrip("/"))
        tok = tokenizer_for(path)
        model = Splade(path).eval()
        n_params = sum(p.numel() for p in model.parameters())

        def make(m, device):
            @torch.no_grad()
            def enc(text):
                b = tok([text], truncation=True, max_length=args.max_len_q, return_tensors="pt").to(device)
                w = m(b["input_ids"], b["attention_mask"]).float()[0]
                if device != "cpu":
                    torch.cuda.synchronize()
                nz = torch.nonzero(w).squeeze(1)
                return nz.cpu().numpy(), w[nz].cpu().numpy()
            return enc

        r = {"model": name, "params_M": round(n_params / 1e6, 1)}
        torch.set_num_threads(1)
        r["cpu1_median_us"], r["cpu1_p95_us"] = time_fn(make(model, "cpu"), timing_texts)
        torch.set_num_threads(4)
        r["cpu4_median_us"], _ = time_fn(make(model, "cpu"), timing_texts)
        qmat = encode_all(make(model, "cpu"), ev.q_texts, ev.vocab)
        if torch.cuda.is_available():
            g = Splade(path).eval().half().to("cuda")
            r["gpu16_median_us"], _ = time_fn(make(g, "cuda"), timing_texts)
            del g
            torch.cuda.empty_cache()
        r.update(ev.quality(qmat))
        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        sp.save_npz(os.path.join(out_dir, "queries.npz"), qmat)
        link = os.path.join(out_dir, "docs.npz")
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(args.enc, "docs.npz")), link)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)

    print("\n=== Кодировщики запроса на одном индексе (время одного запроса, мкс) ===")
    print(f"{'модель':<40} {'парам., М':>9} {'CPU 1п':>8} {'CPU 4п':>8} {'GPU fp16':>9} {'MRR@10':>7} "
          f"{'top-10 как у учителя':>21} {'термов':>7}")
    for r in results:
        g = f"{r['gpu16_median_us']:>9.0f}" if "gpu16_median_us" in r else f"{'—':>9}"
        print(f"{r['model']:<40} {r['params_M']:>9.1f} {r['cpu1_median_us']:>8.0f} {r['cpu4_median_us']:>8.0f} {g} "
              f"{r['MRR@10']:>7.4f} {r['overlap@10_vs_ref']:>21.3f} {r['nnz_q']:>7.1f}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
