#!/bin/bash
# Сборка пропатченного SEISMIC (с search_adaptive) и сравнение с обычным SEISMIC на MS MARCO 200k.
# Запуск из /data/$USER, где лежат seismic_patched.tar.gz и seismic_adaptive_eval.py:
#   nohup bash build_and_run_adaptive.sh > adaptive.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-/data/$USER}
ENC=${ENC:-$DATA/msmarco_200000_enc}
PY311=$DATA/env311/bin/python
export RUSTUP_HOME=$DATA/rustup CARGO_HOME=$DATA/cargo   # Rust в /data, а не в домашней папке
export PIP_CACHE_DIR=$DATA/pipcache

echo "=== 1. Rust (nightly — его требует зависимость vectorium) ==="
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain nightly --no-modify-path || exit 1
fi
source "$CARGO_HOME/env"
rustc --version

echo "=== 2. Сборка колеса ==="
rm -rf seismic_patched wheels_patched
tar xzf seismic_patched.tar.gz
$PY311 -m pip install -q maturin || exit 1
build() { (cd seismic_patched && RUSTUP_TOOLCHAIN=$1 $DATA/env311/bin/maturin build --release -i $PY311 -o ../wheels_patched); }
if ! build nightly; then
  echo "свежий nightly не собрал, пробую nightly-2026-10-15 (около даты выпуска пакета)"
  rustup toolchain install nightly-2026-10-15 --profile minimal && build nightly-2026-10-15 || exit 1
fi

echo "=== 3. Установка в env311 (заменяет обычный pyseismic-lsr; исходный search работает как раньше) ==="
$PY311 -m pip install --force-reinstall --no-deps wheels_patched/*.whl || exit 1
$PY311 -c "import seismic; assert hasattr(seismic.SeismicIndex, 'search_adaptive'); print('search_adaptive ok')" || exit 1

echo "=== 4. Замеры ==="
for NP in 1000 2000; do
  echo "######## n_postings=$NP ########"
  $PY311 seismic_adaptive_eval.py --docs "$ENC/docs.npz" --queries "$ENC/queries.npz" --n-postings $NP \
    --out "$DATA/adapt_$NP.npz" --build-threads 8 | tee adaptive_np$NP.txt
done
echo "готово: adaptive_np*.txt"
