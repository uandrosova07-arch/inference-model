#!/bin/bash
# Ускорение кодировщика запроса: (3) обрезка словаря головы, (4) выборочное int8, (5) кэш, (6) дистилляция
# маленького кодировщика запроса под готовый индекс.
# Запуск из /data/$USER:   GPU=1 nohup bash run_speed.sh > speed.log 2>&1 &
# Если ещё идёт run_if.sh — сначала дождётся его (иначе замеры времени мешали бы друг другу).
set -uo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-/data/$USER}
GPU=${GPU:-1}
ENC=$DATA/msmarco_200000_enc
SUB=$DATA/msmarco_200000
ZIP=$DATA/msmarco.zip
PY=$DATA/env/bin/python
PY311=$DATA/env311/bin/python
TEACHER=naver/splade-cocondenser-ensembledistil
export HF_HOME=$DATA/hf

while pgrep -f "bash run_if.sh" > /dev/null; do echo "жду окончания run_if.sh..."; sleep 300; done

echo "######## 3-4. обрезка словаря и int8 ########"
CUDA_VISIBLE_DEVICES=$GPU $PY speed_tricks.py --model $TEACHER --enc "$ENC" --sub "$SUB" --zip "$ZIP" \
  --out speed_tricks.json | tee speed_tricks.txt

echo "######## 5. кэш ########"
CUDA_VISIBLE_DEVICES=$GPU $PY cache_eval.py --model $TEACHER --enc "$ENC" --sub "$SUB" --zip "$ZIP" \
  --out cache.json | tee cache.txt

echo "######## 6. дистилляция: данные ########"
[ -f "$DATA/distill_data/cand.npy" ] || CUDA_VISIBLE_DEVICES=$GPU $PY distill_prep.py --zip "$ZIP" --enc "$ENC" \
  --sub "$SUB" --out-dir "$DATA/distill_data" --n 100000 > distill_prep.log 2>&1 || { echo "distill_prep упал"; exit 1; }

MODELS=($TEACHER)
for S in L-2_H-128_A-2 L-4_H-256_A-4 L-4_H-512_A-8; do
  OUT=$DATA/stud_$S
  if [ ! -f "$OUT/config.json" ]; then
    echo "######## 6. дистилляция: ученик $S ########"
    CUDA_VISIBLE_DEVICES=$GPU $PY distill_train.py --student google/bert_uncased_$S --data "$DATA/distill_data" \
      --docs "$ENC/docs.npz" --out "$OUT" --steps 6000 > train_stud_$S.log 2>&1 || { echo "ученик $S упал, см. train_stud_$S.log"; continue; }
  fi
  MODELS+=("$OUT")
done

echo "######## 6. дистилляция: оценка ########"
CUDA_VISIBLE_DEVICES=$GPU $PY distill_eval.py --models "${MODELS[@]}" --enc "$ENC" --sub "$SUB" --zip "$ZIP" \
  --out-root "$DATA/stud_enc" --out distill_eval.json | tee distill_eval.txt

echo "######## 6. дистилляция: поиск SEISMIC по векторам учеников ########"
for D in "$DATA"/stud_enc/*/; do
  N=$(basename "$D")
  $PY311 e2e_search.py --enc "$D" --sub "$SUB" --zip "$ZIP" --n-postings 2000 --n-time 1000 \
    --out "search_$N.json" > /dev/null 2>&1 || { echo "поиск для $N упал"; continue; }
  $PY - "$N" <<'EOF'
import json, sys
n = sys.argv[1]
r = next(x for x in json.load(open("distill_eval.json")) if x["model"] == n)
enc = [{"variant": f"{n[:10]} CPU 1п", "median_us": r["cpu1_median_us"], "p95_us": r["cpu1_p95_us"]},
       {"variant": f"{n[:10]} CPU 4п", "median_us": r["cpu4_median_us"], "p95_us": r["cpu4_median_us"]}]
if "gpu16_median_us" in r:
    enc.append({"variant": f"{n[:10]} GPU fp16", "median_us": r["gpu16_median_us"], "p95_us": r["gpu16_median_us"]})
json.dump(enc, open(f"enc_{n}.json", "w"))
EOF
  echo "=== $N ==="
  $PY e2e_report.py --encode "enc_$N.json" --search "search_$N.json"
done | tee distill_search.txt
echo "готово: speed_tricks.txt cache.txt distill_eval.txt distill_search.txt"
