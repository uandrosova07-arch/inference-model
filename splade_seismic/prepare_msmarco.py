"""Подвыборка MS MARCO passage из BEIR: dev-запросы + все их релевантные пассажи + случайные пассажи до --n-docs.

Пишет в --out-dir: corpus.jsonl, queries.jsonl (формат BEIR) — их понимает encode_corpus.py.
Архив (~1 ГБ) скачивается один раз в --zip; корпус читается потоково, без распаковки на диск.
Адрес архива взят по памяти (то же расположение, что у SciFact, которое сработало).

  python prepare_msmarco.py --zip /data/$USER/msmarco.zip --out-dir msmarco_200k --n-docs 200000 --n-queries 2000
"""
import argparse
import json
import os
import random
import urllib.request
import zipfile

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="msmarco.zip", help="куда скачать архив (или где он уже лежит)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-docs", type=int, default=200000)
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.zip):
        print("скачиваю", URL)
        urllib.request.urlretrieve(URL, args.zip)
    os.makedirs(args.out_dir, exist_ok=True)
    rnd = random.Random(args.seed)

    with zipfile.ZipFile(args.zip) as z:
        names = z.namelist()
        find = lambda suffix: next(n for n in names if n.endswith(suffix))

        # dev-запросы и их релевантные пассажи
        qrels = {}
        with z.open(find("qrels/dev.tsv")) as f:
            next(f)  # заголовок query-id corpus-id score
            for line in f:
                qid, did, score = line.decode("utf-8").rstrip("\n").split("\t")
                if int(score) > 0:
                    qrels.setdefault(qid, set()).add(did)
        qids = sorted(qrels)
        rnd.shuffle(qids)
        qids = set(qids[:args.n_queries])
        relevant = set().union(*(qrels[q] for q in qids))
        if len(relevant) > args.n_docs:
            raise SystemExit(f"релевантных пассажей {len(relevant)} больше, чем --n-docs")

        n_rand = args.n_docs - len(relevant)
        picked_rel, reservoir, seen = [], [], 0
        with z.open(find("corpus.jsonl")) as f:
            for line in f:
                did = json.loads(line)["_id"]
                if did in relevant:
                    picked_rel.append(line)
                    continue
                seen += 1  # резервуарная выборка: равномерно n_rand из всех нерелевантных
                if len(reservoir) < n_rand:
                    reservoir.append(line)
                else:
                    j = rnd.randrange(seen)
                    if j < n_rand:
                        reservoir[j] = line
        docs = picked_rel + reservoir
        rnd.shuffle(docs)
        with open(os.path.join(args.out_dir, "corpus.jsonl"), "wb") as out:
            out.writelines(docs)

        n_q = 0
        with z.open(find("queries.jsonl")) as f, open(os.path.join(args.out_dir, "queries.jsonl"), "wb") as out:
            for line in f:
                if json.loads(line)["_id"] in qids:
                    out.write(line)
                    n_q += 1

    print(f"готово: {len(docs)} пассажей (из них релевантных {len(picked_rel)}), {n_q} запросов; "
          f"всего в корпусе {seen + len(picked_rel)}")


if __name__ == "__main__":
    main()
