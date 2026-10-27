"""Причинная проверка: зависит ли запас для адаптивного поиска в SEISMIC от «остроты» весов запроса.

Веса каждого запроса возводятся в степень p (документы и индекс не меняются): p > 1 делает веса острее
(главные термы доминируют), p < 1 — ровнее. Для каждого p:
  * качество при точном поиске: MRR@10 и совпадение top-10 с исходными запросами (p = 1);
  * форма весов: доля главного терма, доля трёх главных, число термов;
  * seismic_eval: оракульный запас (ratio oracle) и линейный предсказатель;
  * seismic_adaptive_eval: выигрыш остановки по термам (патч) с 95% CI.
Гипотеза подтверждается, если при росте p запас и выигрыш патча падают (на inference-free), а при p < 1
на SPLADE — растут. Сетки параметров уменьшены, чтобы уложиться в ~15 мин на одно p; одинаковы для всех p.

  python sharpen_eval.py --enc ENC_IF --sub SUB --zip msmarco.zip --name if --p 0.5 1 2 3 --out-root sharpen
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
import scipy.sparse as sp

import seismic_adaptive_eval as sae
import seismic_eval as se
from qeval import Evaluator

se.CUTS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
se.HFS = [0.8, 0.9, 0.95, 1.0]
sae.CUTS, sae.HFS = se.CUTS, se.HFS
sae.A_CUTS = [8, 15, 20, 30]
sae.A_HFS = [0.9, 0.95, 1.0]
sae.A_EPS = [0.001, 0.01, 0.05]
sae.A_MIN = [2, 3]

ROW = re.compile(r"^\s+(0\.\d+) \| (\w+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+|nan)(?:\s+\[([\d.]+), ([\d.]+)\])?")


def run_main(module, argv, log_path):
    buf = io.StringIO()
    sys.argv = argv
    with contextlib.redirect_stdout(buf):
        module.main()
    text = buf.getvalue()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    rows = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            t, meth, rec, us, p95, ratio, lo, hi = m.groups()
            rows[f"{t}/{meth}"] = {"recall": float(rec), "us": int(us), "p95": int(p95), "ratio": float(ratio),
                                   "ci": [float(lo), float(hi)] if lo else None}
    return rows


def shape(q):
    top1, top3, n = [], [], []
    for i in range(q.shape[0]):
        v = np.sort(q.data[q.indptr[i]:q.indptr[i + 1]])[::-1]
        if v.size:
            top1.append(v[0] / v.sum()); top3.append(v[:3].sum() / v.sum()); n.append(v.size)
    return {"top1_share": round(float(np.mean(top1)), 3), "top3_share": round(float(np.mean(top3)), 3),
            "terms": round(float(np.mean(n)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--p", type=float, nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-postings", type=int, default=2000)
    args = ap.parse_args()

    ev = Evaluator(args.enc, args.sub, args.zip)
    q0 = ev.q_ref
    results = []
    for p in args.p:
        tag = f"{args.name}_p{p:g}"
        d = os.path.join(args.out_root, tag)
        os.makedirs(d, exist_ok=True)
        q = q0.copy()
        q.data = q.data.astype(np.float64) ** p
        q = q.astype(np.float32)
        sp.save_npz(os.path.join(d, "queries.npz"), q)
        r = {"name": args.name, "p": p}
        r.update(shape(q))
        r.update({k: v for k, v in ev.quality(q).items() if k != "nnz_q"})
        print(f"[{tag}] форма и качество: {json.dumps(r, ensure_ascii=False)}", flush=True)
        docs = os.path.join(args.enc, "docs.npz")
        qp = os.path.join(d, "queries.npz")
        r["seismic"] = run_main(se, ["seismic_eval.py", "--docs", docs, "--queries", qp, "--n-postings",
                                     str(args.n_postings), "--out", os.path.join(d, "grid.npz")],
                                os.path.join(d, "seismic_eval.txt"))
        print(f"[{tag}] seismic_eval готов", flush=True)
        r["adapt"] = run_main(sae, ["seismic_adaptive_eval.py", "--docs", docs, "--queries", qp, "--n-postings",
                                    str(args.n_postings), "--out", os.path.join(d, "adapt.npz")],
                              os.path.join(d, "adaptive.txt"))
        print(f"[{tag}] adaptive готов", flush=True)
        results.append(r)
        with open(os.path.join(args.out_root, f"{args.name}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

    print(f"\n=== {args.name}: веса запроса в степени p (документы те же) ===")
    print(f"{'p':>4} {'доля top1':>9} {'доля top3':>9} {'MRR@10':>7} {'top-10 как p=1':>14} | "
          f"{'оракул 0.90':>11} {'оракул 0.95':>11} {'linear 0.90':>11} | {'патч 0.90':>16} {'патч 0.95':>16} "
          f"{'патч 0.97':>16} | {'база 0.95, мкс':>14}")
    for r in results:
        s, a = r["seismic"], r["adapt"]
        g = lambda dct, key: f"{dct[key]['ratio']:.2f}" if key in dct else "—"

        def ci(key):
            if key not in a:
                return "—"
            x = a[key]
            return f"{x['ratio']:.2f} [{x['ci'][0]:.2f},{x['ci'][1]:.2f}]" if x["ci"] else f"{x['ratio']:.2f}"
        base_us = a.get("0.95/base", {}).get("us", "—")
        print(f"{r['p']:>4g} {r['top1_share']:>9.3f} {r['top3_share']:>9.3f} {r['MRR@10']:>7.4f} "
              f"{r['overlap@10_vs_ref']:>14.3f} | {g(s, '0.90/oracle'):>11} {g(s, '0.95/oracle'):>11} "
              f"{g(s, '0.90/linear'):>11} | {ci('0.90/adapt'):>16} {ci('0.95/adapt'):>16} {ci('0.97/adapt'):>16} | "
              f"{base_us:>14}")
    print("\nоракул/linear/патч — доля задержки лучшей общей настройки при той же полноте (меньше — больше выигрыш).")


if __name__ == "__main__":
    main()
