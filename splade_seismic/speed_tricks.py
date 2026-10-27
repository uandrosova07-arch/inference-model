"""Ускорение кодировщика запроса SPLADE без обучения: обрезка словаря MLM-головы и выборочное int8-квантование.

Обрезка словаря. Голова считает логиты для всех 30 522 токенов, но токен, которого нет ни в одном документе
индекса, не может изменить ни один скор: его вес в запросе умножается на ноль. Поэтому при keep = {токены с df >= 1}
выдача обязана совпасть с исходной ТОЧНО (это проверяется: top-10 как у исходных = 1.000). Более жёсткие пороги
(df >= 10, 100, 1000) уже приближённые — для них меряется потеря.
Квантование (динамическое int8, CPU): всё сразу (как в e2e_encode.py), только энкодер, только FFN-слои, только голова,
плюс сочетание «обрезка df>=1 + int8 энкодера».
Время — один запрос, один поток CPU (и GPU fp16, batch=1, если есть); качество — на всех запросах (4 потока).

  python speed_tricks.py --model naver/splade-cocondenser-ensembledistil --enc ENC --sub SUB --zip msmarco.zip
"""
import argparse
import copy
import json
import warnings

import numpy as np
import torch
import torch.nn as nn

from qeval import Evaluator, encode_all, time_fn
from splade import Splade, tokenizer_for


class FastSplade(nn.Module):
    """SPLADE, у которого голова считает логиты только для токенов keep (None — для всех).

    forward возвращает B x len(keep); индексы токенов — self.keep.
    """

    def __init__(self, splade, keep=None):
        super().__init__()
        mlm = splade.mlm
        self.encoder = getattr(mlm, mlm.base_model_prefix)
        self.transform = mlm.cls.predictions.transform
        dec = mlm.cls.predictions.decoder
        vocab = dec.weight.shape[0]
        keep = np.arange(vocab) if keep is None else np.asarray(keep)
        idx = torch.as_tensor(keep, dtype=torch.long)
        self.decoder = nn.Linear(dec.weight.shape[1], len(keep))
        with torch.no_grad():
            self.decoder.weight.copy_(dec.weight[idx])
            self.decoder.bias.copy_(dec.bias[idx])
        self.keep = keep

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        z = self.decoder(self.transform(h))
        return (torch.log1p(torch.relu(z)) * attention_mask.unsqueeze(-1).to(z.dtype)).max(dim=1).values


def int8(model, which):
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    pick = {
        "all": names,
        "encoder": [n for n in names if n.startswith("encoder.")],
        "ffn": [n for n in names if n.endswith("intermediate.dense") or
                (n.endswith("output.dense") and "attention" not in n)],
        "head": [n for n in names if not n.startswith("encoder.")],
    }[which]
    spec = {n: torch.quantization.default_dynamic_qconfig for n in pick}
    return torch.quantization.quantize_dynamic(copy.deepcopy(model), spec, dtype=torch.qint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n-time", type=int, default=300)
    ap.add_argument("--max-len-q", type=int, default=64)
    ap.add_argument("--df-thresholds", type=int, nargs="+", default=[1, 10, 100, 1000])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    ev = Evaluator(args.enc, args.sub, args.zip)
    import scipy.sparse as sp
    docs = sp.load_npz(f"{args.enc}/docs.npz").tocsr()
    df = np.bincount(docs.indices, minlength=ev.vocab)
    del docs
    tok = tokenizer_for(args.model)
    base = Splade(args.model).eval()
    timing_texts = ev.q_texts[:args.n_time]
    results = []

    def encoder(model, device="cpu", half=False):
        keep_t = torch.as_tensor(model.keep)

        @torch.no_grad()
        def enc(text):
            b = tok([text], truncation=True, max_length=args.max_len_q, return_tensors="pt").to(device)
            w = model(b["input_ids"], b["attention_mask"]).float()[0]
            if device != "cpu":
                torch.cuda.synchronize()
            nz = torch.nonzero(w).squeeze(1)
            return keep_t[nz.cpu()].numpy(), w[nz].cpu().numpy()
        return enc

    def record(name, model, extra=None, device="cpu", half=False):
        enc = encoder(model, device, half)
        if device == "cpu":
            torch.set_num_threads(1)
        med, p95 = time_fn(enc, timing_texts)
        r = {"variant": name, "median_us": med, "p95_us": p95}
        if extra:
            r.update(extra)
        torch.set_num_threads(4)
        r.update(ev.quality(encode_all(enc, ev.q_texts, ev.vocab)))
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)

    full = FastSplade(base).eval()
    record("fp32", full, {"vocab": ev.vocab})
    for t in args.df_thresholds:
        keep = np.nonzero(df >= t)[0]
        record(f"prune-df>={t}", FastSplade(base, keep).eval(), {"vocab": int(len(keep))})
    for which in ("all", "encoder", "ffn", "head"):
        record(f"int8-{which}", int8(full, which))
    keep1 = np.nonzero(df >= 1)[0]
    record("prune-df>=1+int8-encoder", int8(FastSplade(base, keep1).eval(), "encoder"), {"vocab": int(len(keep1))})

    if torch.cuda.is_available():
        for name, keep in (("gpu-fp16", None), ("gpu-fp16-prune-df>=1", keep1)):
            m = FastSplade(base, keep).eval().half().to("cuda")
            record(name, m, {"vocab": int(len(m.keep))}, device="cuda")
            del m
            torch.cuda.empty_cache()

    print("\n=== Кодирование запроса без обучения (CPU — 1 поток; один запрос, мкс) ===")
    print(f"{'вариант':<26} {'словарь':>8} {'медиана':>9} {'p95':>9} {'ускорение':>9} {'MRR@10':>7} "
          f"{'top-10 как у исх.':>17} {'термов':>7}")
    ref = {"cpu": results[0]["median_us"]}
    for r in results:
        gpu = r["variant"].startswith("gpu")
        base_t = next((x["median_us"] for x in results if x["variant"] == "gpu-fp16"), None) if gpu else ref["cpu"]
        sp_ = f"{base_t / r['median_us']:>8.2f}x" if base_t else f"{'—':>9}"
        voc = f"{r['vocab']:>8}" if "vocab" in r else f"{'':>8}"
        print(f"{r['variant']:<26} {voc} {r['median_us']:>9.0f} {r['p95_us']:>9.0f} {sp_} {r['MRR@10']:>7.4f} "
              f"{r['overlap@10_vs_ref']:>17.3f} {r['nnz_q']:>7.1f}")
    print("\nprune-df>=1 обязан дать top-10 как у исходных = 1.000 (точная обрезка); для GPU ускорение — "
          "относительно gpu-fp16. Учтите: на полном MS MARCO (8,8 млн) с df>=1 останется больше токенов, чем на 200k.")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
