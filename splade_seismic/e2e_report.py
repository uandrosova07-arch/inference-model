"""Сквозное время, часть 3: сводка из e2e_encode.json и e2e_search.json.

  python e2e_report.py --encode e2e_encode.json --search e2e_search.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", required=True)
    ap.add_argument("--search", required=True)
    args = ap.parse_args()
    enc = json.load(open(args.encode, encoding="utf-8"))
    srch = json.load(open(args.search, encoding="utf-8"))

    print("=== Кодирование запроса (один запрос, мкс) ===")
    print(f"{'вариант':<16} {'медиана':>9} {'p95':>9} {'MRR@10':>7} {'top-10 как у исходных':>22}")
    for r in enc:
        q = f"{r['MRR@10']:>7.4f} {r['overlap@10_vs_ref']:>22.3f}" if "MRR@10" in r else f"{'—':>7} {'—':>22}"
        print(f"{r['variant']:<16} {r['median_us']:>9.0f} {r['p95_us']:>9.0f} {q}")

    print("\n=== Поиск в SEISMIC, включая досчёт кандидатов (один поток, мкс) ===")
    print(f"{'настройка':<16} {'медиана':>9} {'p95':>9} {'recall@10':>9} {'MRR@10':>7}")
    for r in srch:
        print(f"{r['config']:<16} {r.get('median_us', float('nan')):>9.0f} {r.get('p95_us', float('nan')):>9.0f} "
              f"{r['recall@10']:>9.3f} {r['MRR@10']:>7.4f}")

    picks = {}
    for target in (0.90, 0.95, 0.97):
        ok = [r for r in srch if r["config"] != "exact" and r["recall@10"] >= target]
        if ok:
            picks[target] = min(ok, key=lambda r: r["median_us"])
    if not picks:  # ни одна настройка не дотянула до 0,90 — берём самую полную
        best = max((r for r in srch if r["config"] != "exact"), key=lambda r: r["recall@10"])
        picks[best["recall@10"]] = best
    print("\n=== Сквозное время = кодирование + поиск (медианы, мкс); доля кодирования ===")
    head = "".join(f" | полнота≥{t:.2f} ({p['config']})" for t, p in picks.items())
    print(f"{'кодировщик':<16}{head}")
    for r in enc:
        cells = ""
        for t, p in picks.items():
            total = r["median_us"] + p["median_us"]
            cells += f" | {total:>9.0f} мкс, кодирование {100 * r['median_us'] / total:>4.1f}%".ljust(len(f" | полнота≥{t:.2f} ({p['config']})"))
        print(f"{r['variant']:<16}{cells}")
    if any(r["variant"] in ("cpu-L4-1t", "cpu-L2-1t", "inference-free") for r in enc):
        print("\nL2/L4 и inference-free — нижние границы времени без обученной модели; их качество не измерялось.")


if __name__ == "__main__":
    main()
