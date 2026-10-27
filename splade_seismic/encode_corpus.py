"""РљРѕРґРёСЂСѓРµС‚ РєРѕСЂРїСѓСЃ Рё Р·Р°РїСЂРѕСЃС‹ (С„РѕСЂРјР°С‚ BEIR jsonl) РјРѕРґРµР»СЊСЋ SPLADE РІ scipy CSR: docs.npz Рё queries.npz.

РџСЂРёРјРµСЂ (РјРѕРґРµР»СЊ РёР· СЃС‚Р°С‚СЊРё SEISMIC):
  python encode_corpus.py --model naver/splade-cocondenser-ensembledistil \
      --corpus scifact/corpus.jsonl --queries scifact/queries.jsonl --out-dir scifact_enc
"""
import argparse
import json
import os

import scipy.sparse as sp
import torch

from splade import Splade, encode_to_csr, tokenizer_for


def read_jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-len-d", type=int, default=256)
    ap.add_argument("--max-len-q", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true", help="autocast fp16 РЅР° GPU (РїСЂРёРјРµСЂРЅРѕ РІРґРІРѕРµ Р±С‹СЃС‚СЂРµРµ)")
    ap.add_argument("--log-every", type=int, default=500, help="РїРµС‡Р°С‚Р°С‚СЊ РїСЂРѕРіСЂРµСЃСЃ РєР°Р¶РґС‹Рµ N Р±Р°С‚С‡РµР№ (0 вЂ” РјРѕР»С‡Р°С‚СЊ)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = tokenizer_for(args.model)
    model = Splade(args.model).to(args.device)

    docs = read_jsonl(args.corpus, args.max_docs)
    queries = read_jsonl(args.queries, args.max_queries)
    doc_texts = [(d.get("title", "") + " " + d["text"]).strip() for d in docs]
    query_texts = [q["text"] for q in queries]

    print(f"РєРѕРґРёСЂСѓСЋ {len(doc_texts)} РґРѕРєСѓРјРµРЅС‚РѕРІ РЅР° {args.device}", flush=True)
    d_mat = encode_to_csr(model, tok, doc_texts, args.max_len_d, args.batch_size, args.device,
                          fp16=args.fp16, log_every=args.log_every)
    q_mat = encode_to_csr(model, tok, query_texts, args.max_len_q, args.batch_size, args.device,
                          fp16=args.fp16)
    sp.save_npz(os.path.join(args.out_dir, "docs.npz"), d_mat)
    sp.save_npz(os.path.join(args.out_dir, "queries.npz"), q_mat)
    print(f"docs {d_mat.shape}, СЃСЂРµРґРЅРµРµ С‡РёСЃР»Рѕ РЅРµРЅСѓР»РµРІС‹С… {d_mat.nnz / d_mat.shape[0]:.1f}; "
          f"queries {q_mat.shape}, СЃСЂРµРґРЅРµРµ С‡РёСЃР»Рѕ РЅРµРЅСѓР»РµРІС‹С… {q_mat.nnz / q_mat.shape[0]:.1f}")


if __name__ == "__main__":
    main()
