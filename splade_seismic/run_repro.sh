#!/bin/bash
# Проверка воспроизводимости выигрыша адаптивного SEISMIC: настройки измеряются в случайном порядке
# (обычные и адаптивные вперемешку) с контролем дрейфа нагрузки; inference-free — два независимых прогона.
# Запуск из /data/$USER:   nohup bash run_repro.sh > repro.log 2>&1 &      (около 4 часов)
set -uo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-/data/$USER}
SUB=$DATA/msmarco_200000
ZIP=$DATA/msmarco.zip
PY=$DATA/env/bin/python
PY311=$DATA/env311/bin/python

while pgrep -f "bash run_(if|speed|sharpen|pareto).sh" > /dev/null; do echo "жду окончания других прогонов..."; sleep 300; done

for R in 1 2; do
  echo "######## inference-free, прогон $R ########"
  $PY311 seismic_adaptive_eval.py --docs "$DATA/msmarco_200000_if/docs.npz" --queries "$DATA/msmarco_200000_if/queries.npz" \
    --n-postings 2000 --out "$DATA/adapt_if_r$R.npz" > adaptive_if_r$R.txt 2>&1
  grep -E "дрейф|^ +0\.9" adaptive_if_r$R.txt
done

echo "######## SPLADE ########"
$PY311 seismic_adaptive_eval.py --docs "$DATA/msmarco_200000_enc/docs.npz" --queries "$DATA/msmarco_200000_enc/queries.npz" \
  --n-postings 2000 --out "$DATA/adapt_splade_r1.npz" > adaptive_splade_r1.txt 2>&1
grep -E "дрейф|^ +0\.9" adaptive_splade_r1.txt

echo "######## inference-free: кривые скорость — качество (случайный порядок) ########"
$PY311 pareto_eval.py --docs "$DATA/msmarco_200000_if/docs.npz" --queries "$DATA/msmarco_200000_if/queries.npz" \
  --sub "$SUB" --zip "$ZIP" --out "$DATA/pareto_if_v2.npz" --report pareto_if_v2.txt --name inference-free \
  > pareto_if_v2_measure.log 2>&1
$PY pareto_eval.py --docs "$DATA/msmarco_200000_if/docs.npz" --queries "$DATA/msmarco_200000_if/queries.npz" \
  --sub "$SUB" --zip "$ZIP" --out "$DATA/pareto_if_v2.npz" --report pareto_if_v2.txt --name inference-free \
  --analyze-only > /dev/null 2>&1
grep "дрейф" pareto_if_v2_measure.log
echo "готово: adaptive_if_r1.txt adaptive_if_r2.txt adaptive_splade_r1.txt pareto_if_v2.txt pareto_if_v2.png"
