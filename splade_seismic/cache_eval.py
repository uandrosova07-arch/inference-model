"""Кэш векторов запросов: сколько запросов dev уже встречались в логе (train-запросы MS MARCO) и что даёт кэш.

Ключи кэша, от строгого к мягкому:
  exact — текст как есть; norm — нижний регистр, без пунктуации, схлопнутые пробелы;
  bag   — отсортированное множество слов после norm (порядок слов игнорируется — это уже приближение:
          SPLADE зависит от порядка, поэтому для таких попаданий меряется, насколько меняется выдача).
Для каждого ключа: доля попаданий, время поиска в кэше, MRR@10 и совпадение top-10, если для попаданий взять
вектор кэшированного train-запроса вместо собственного.
Оговорка: MS MARCO собран из дедуплицированных запросов, поэтому доля повторов тут — НИЖНЯЯ оценка; в настоящих
логах поисковиков повторяется заметная часть потока, и тогда кэш — самое дешёвое ускорение из всех.

  python cache_eval.py --model naver/splade-cocondenser-ensembledistil --enc ENC --sub SUB --zip msmarco.zip
"""
import argparse
import json
import re
import time
import zipfile

import numpy as np
import torch

from qeval import Evaluator
from splade import Splade, encode_to_csr, tokenizer_for

PUNCT = re.compile(r"[^\w\s]")


def key_exact(t):
    return t


def key_norm(t):
    return " ".join(PUNCT.sub(" ", t.lower()).split())


def key_bag(t):
    return " ".join(sorted(set(key_norm(t).split())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ev = Evaluator(args.enc, args.sub, args.zip)
    dev_ids = set(ev.q_ids)
    with zipfile.ZipFile(args.zip) as z:
        with z.open(next(n for n in z.namelist() if n.endswith("qrels/train.tsv"))) as f:
            next(f)
            train_ids = {line.decode("utf-8").split("\t")[0] for line in f}
        with z.open(next(n for n in z.namelist() if n.endswith("queries.jsonl"))) as f:
            log = [json.loads(line) for line in f]
    log = [q["text"] for q in log if q["_id"] in train_ids and q["_id"] not in dev_ids]
    print(f"лог (train-запросы): {len(log)}; запросов dev: {len(ev.q_texts)}", flush=True)

    tok = tokenizer_for(args.model)
    model = Splade(args.model).to(args.device).eval()
    results = []
    for name, key in (("exact", key_exact), ("norm", key_norm), ("bag", key_bag)):
        cache = {}
        for t in log:
            cache.setdefault(key(t), t)
        lat, hits = [], {}
        for qi, t in enumerate(ev.q_texts):
            t0 = time.perf_counter_ns()
            hit = cache.get(key(t))
            lat.append((time.perf_counter_ns() - t0) / 1000.0)
            if hit is not None:
                hits[qi] = hit
        r = {"key": name, "cache_size": len(cache), "hit_rate": round(len(hits) / len(ev.q_texts), 4),
             "lookup_median_us": round(float(np.median(lat)), 2)}
        if hits:
            idx = sorted(hits)
            cached = encode_to_csr(model, tok, [hits[i] for i in idx], 64, 64, args.device, fp16=True).tolil()
            q = ev.q_ref.tolil(copy=True)
            for j, i in enumerate(idx):
                q[i] = cached[j]
            q = q.tocsr()
            sub_ref = [ev.ref_top[i] for i in idx]
            tops = ev.top_k(q[idx])
            r["overlap@10_on_hits"] = round(float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(tops, sub_ref)])), 4)
            r.update({f"{k}_with_cache": v for k, v in ev.quality(q).items() if k != "nnz_q"})
            r["examples"] = [(ev.q_texts[i], hits[i]) for i in idx[:5]]
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)

    print("\n=== Кэш векторов запросов (лог = train-запросы MS MARCO) ===")
    print(f"{'ключ':<6} {'размер кэша':>12} {'попадания':>10} {'поиск в кэше, мкс':>18} "
          f"{'top-10 на попаданиях':>21} {'MRR@10 с кэшем':>15}")
    for r in results:
        ov = f"{r['overlap@10_on_hits']:>21.3f}" if "overlap@10_on_hits" in r else f"{'—':>21}"
        mrr = f"{r['MRR@10_with_cache']:>15.4f}" if "MRR@10_with_cache" in r else f"{'—':>15}"
        print(f"{r['key']:<6} {r['cache_size']:>12} {100 * r['hit_rate']:>9.2f}% {r['lookup_median_us']:>18.2f} {ov} {mrr}")
    print("\nСредняя экономия на запрос = доля попаданий x время кодирования; поиск в кэше стоит ~1 мкс.")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
