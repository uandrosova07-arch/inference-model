#!/bin/bash
# Причинная проверка «запас для адаптивного поиска зависит от остроты весов запроса».
# Запуск из /data/$USER:   nohup bash run_sharpen.sh > sharpen.log 2>&1 &
# Ждёт окончания run_speed.sh (замеры времени не должны идти одновременно). Около 1.5–2 часов.
set -uo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-/data/$USER}
SUB=$DATA/msmarco_200000
PY311=$DATA/env311/bin/python

while pgrep -f "bash run_(if|speed).sh" > /dev/null; do echo "жду окончания run_if.sh / run_speed.sh..."; sleep 300; done
$PY311 -c "import seismic; assert hasattr(seismic.SeismicIndex, 'search_adaptive')" || {
  echo "нужна пропатченная сборка SEISMIC (search_adaptive)"; exit 1; }

echo "######## inference-free: веса острее (p > 1) и ровнее (p < 1) ########"
$PY311 sharpen_eval.py --enc "$DATA/msmarco_200000_if" --sub "$SUB" --zip "$DATA/msmarco.zip" --name if \
  --p 1 2 3 0.5 --out-root "$DATA/sharpen" > sharpen_if.log 2>&1
tail -12 sharpen_if.log | tee sharpen_if.txt

echo "######## SPLADE: веса ровнее (p < 1) ########"
$PY311 sharpen_eval.py --enc "$DATA/msmarco_200000_enc" --sub "$SUB" --zip "$DATA/msmarco.zip" --name splade \
  --p 1 0.5 0.3 --out-root "$DATA/sharpen" > sharpen_splade.log 2>&1
tail -11 sharpen_splade.log | tee sharpen_splade.txt
echo "готово: sharpen_if.txt sharpen_splade.txt"
