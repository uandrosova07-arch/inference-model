#!/bin/bash
# Сквозное время (кодирование запроса + поиск SEISMIC) и проверка корреляции глубины с концентрацией документов.
# Запуск из /data/$USER:   GPU=1 nohup bash run_e2e.sh > e2e.log 2>&1 &
# Всё по очереди: замеры времени не должны идти параллельно с другими нашими задачами.
set -uo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-/data/$USER}
GPU=${GPU:-1}
ENC=$DATA/msmarco_200000_enc
SUB=$DATA/msmarco_200000
export HF_HOME=$DATA/hf

echo "######## 1. кодирование запроса ########"
CUDA_VISIBLE_DEVICES=$GPU $DATA/env/bin/python e2e_encode.py --model naver/splade-cocondenser-ensembledistil \
  --enc "$ENC" --sub "$SUB" --zip "$DATA/msmarco.zip" --n-time 500 --out e2e_encode.json || exit 1

echo "######## 2. поиск SEISMIC ########"
$DATA/env311/bin/python e2e_search.py --enc "$ENC" --sub "$SUB" --zip "$DATA/msmarco.zip" --n-postings 2000 \
  --n-time 1000 --out e2e_search.json || exit 1

echo "######## 3. сводка ########"
$DATA/env/bin/python e2e_report.py --encode e2e_encode.json --search e2e_search.json | tee e2e_report.txt

echo "######## 4. корреляция глубины и концентрации документов ########"
$DATA/env/bin/python corr_doc_conc.py --docs "$ENC/docs.npz" --queries "$ENC/queries.npz" --lam 2000 | tee corr_lam2000.txt
echo "готово: e2e_report.txt corr_lam2000.txt"
