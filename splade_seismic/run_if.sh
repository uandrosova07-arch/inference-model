#!/bin/bash
# 1) inference-free модель: кодирование корпуса, сквозное время, запас для ускорения поиска (SEISMIC + симулятор);
# 2) кодирование запросов SPLADE пачками на GPU/CPU.
# Запуск из /data/$USER:   GPU=1 nohup bash run_if.sh > if.log 2>&1 &
# Всё по очереди, чтобы замеры времени не мешали друг другу.
set -uo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-/data/$USER}
GPU=${GPU:-1}
ENC=$DATA/msmarco_200000_enc
ENC_IF=$DATA/msmarco_200000_if
SUB=$DATA/msmarco_200000
PY=$DATA/env/bin/python
PY311=$DATA/env311/bin/python
export HF_HOME=$DATA/hf

echo "######## 1. inference-free: кодирование корпуса (GPU) ########"
if [ ! -f "$ENC_IF/queries.npz" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY encode_if.py --sub "$SUB" --out-dir "$ENC_IF" --timing-out if_encode.json \
    --fp16 --batch-size 64 || exit 1
fi

echo "######## 2. SPLADE: кодирование запросов пачками ########"
CUDA_VISIBLE_DEVICES=$GPU $PY e2e_batch.py --model naver/splade-cocondenser-ensembledistil --enc "$ENC" \
  --sub "$SUB" --out e2e_batch.json | tee e2e_batch.txt

echo "######## 3. inference-free: поиск SEISMIC и сквозное время ########"
$PY311 e2e_search.py --enc "$ENC_IF" --sub "$SUB" --zip "$DATA/msmarco.zip" --n-postings 2000 \
  --n-time 1000 --out e2e_search_if.json || exit 1
$PY e2e_report.py --encode if_encode.json --search e2e_search_if.json | tee e2e_report_if.txt

echo "######## 4. inference-free: запас для адаптивного поиска в настоящем SEISMIC ########"
$PY311 seismic_eval.py --docs "$ENC_IF/docs.npz" --queries "$ENC_IF/queries.npz" --n-postings 2000 \
  --out "$DATA/seismic_if.npz" > seismic_if.txt 2>&1
if $PY311 -c "import seismic; assert hasattr(seismic.SeismicIndex, 'search_adaptive')" 2>/dev/null; then
  $PY311 seismic_adaptive_eval.py --docs "$ENC_IF/docs.npz" --queries "$ENC_IF/queries.npz" --n-postings 2000 \
    --out "$DATA/adapt_if.npz" > adaptive_if.txt 2>&1
else
  echo "пропатченный SEISMIC не установлен — остановку по термам пропускаю" | tee adaptive_if.txt
fi

echo "######## 5. inference-free: симулятор (обход по убыванию вклада, методы внутри поиска) ########"
CUDA_VISIBLE_DEVICES="" $PY eval_depth.py --docs "$ENC_IF/docs.npz" --queries "$ENC_IF/queries.npz" --lam 2000 \
  > depth_if.txt 2>&1
echo "готово: e2e_batch.txt e2e_report_if.txt seismic_if.txt adaptive_if.txt depth_if.txt"
