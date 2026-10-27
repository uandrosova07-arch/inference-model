#!/bin/bash
# Полный прогон на подвыборке MS MARCO: подготовка -> кодирование -> оценка (lam 1000 и 2000, два критерия выбора).
# Запуск (из папки с .py-файлами, лучше внутри tmux):
#   CUDA_VISIBLE_DEVICES=0 bash run_msmarco.sh
# Параметры через переменные окружения: N_DOCS (200000), N_Q (2000), DATA (/data/$USER).
set -euo pipefail
cd "$(dirname "$0")"

DATA=${DATA:-/data/$USER}
N_DOCS=${N_DOCS:-200000}
N_Q=${N_Q:-2000}
export HF_HOME=${HF_HOME:-$DATA/hf}   # кеш модели в /data, а не в home (там квота меньше)
mkdir -p "$DATA" "$HF_HOME"

SUB=$DATA/msmarco_${N_DOCS}
ENC=$DATA/msmarco_${N_DOCS}_enc

if [ ! -f "$SUB/corpus.jsonl" ]; then
  python3 prepare_msmarco.py --zip "$DATA/msmarco.zip" --out-dir "$SUB" --n-docs "$N_DOCS" --n-queries "$N_Q"
fi

if [ ! -f "$ENC/docs.npz" ]; then
  python3 encode_corpus.py --model naver/splade-cocondenser-ensembledistil \
    --corpus "$SUB/corpus.jsonl" --queries "$SUB/queries.jsonl" --out-dir "$ENC" \
    --batch-size 64 --fp16 --max-len-d 256 --max-len-q 64
fi

for LAM in 1000 2000; do
  for SEL in cost cost_p95; do
    echo "######## lam=$LAM select-by=$SEL ########"
    python3 eval_cut_v2.py --docs "$ENC/docs.npz" --queries "$ENC/queries.npz" \
      --k 10 --lam "$LAM" --max-cut 50 --targets 0.90 0.95 0.97 --select-by "$SEL" \
      | tee "results_lam${LAM}_${SEL}.txt"
  done
done
echo "готово: results_lam*.txt"
