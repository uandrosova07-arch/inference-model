#!/bin/bash
# Методы «во время поиска» (eval_depth_2.py) и настоящий SEISMIC (seismic_eval.py) на подвыборке MS MARCO.
# Запуск из /data/$USER:  CUDA_VISIBLE_DEVICES="" nohup bash run_next.sh > next.log 2>&1 &
# Нужны два окружения: env (Python 3.12, основное) и env311 (Python 3.11, для pyseismic-lsr).
set -uo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-/data/$USER}
ENC=${ENC:-$DATA/msmarco_200000_enc}
PY=$DATA/env/bin/python
PY311=$DATA/env311/bin/python

for LAM in 1000 2000; do
  echo "######## eval_depth lam=$LAM ########"
  $PY eval_depth.py --docs "$ENC/docs.npz" --queries "$ENC/queries.npz" --lam $LAM | tee depth2_lam$LAM.txt
done

for NP in 1000 2000; do
  echo "######## SEISMIC n_postings=$NP ########"
  $PY311 seismic_eval.py --docs "$ENC/docs.npz" --queries "$ENC/queries.npz" --n-postings $NP \
    --index "$DATA/seismic_idx_$NP" --out "$DATA/seismic_$NP.npz" --build-threads 8 | tee seismic_np$NP.txt
done
echo "готово: depth2_lam*.txt seismic_np*.txt"
