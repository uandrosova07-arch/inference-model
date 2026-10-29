# Sparse Retrieval Inference Boost

Воспроизводимый экспериментальный pipeline для sparse retrieval:

```text
MS MARCO Passage
    -> SPLADE document/query encoding
    -> SEISMIC indexing and retrieval
    -> quality, latency and artifact reports
```

Основной сценарий — полный baseline на MS MARCO Passage с
`naver/splade-cocondenser-selfdistil`. Pipeline поддерживает две GPU,
возобновление после остановки, повторное использование готовых артефактов и
lineage экспериментов через `manifest.json`.

## Быстрый старт: baseline на Kaggle

Требования к notebook:

- `Accelerator: GPU T4 x2`;
- `Internet: On` для clone, установки пакетов и первой загрузки модели;
- подключённый датасет
  [`iskulovdamir/msmarco`](https://www.kaggle.com/datasets/iskulovdamir/msmarco).

```bash
!git clone --depth 1 --branch damir_branch https://github.com/uandrosova07-arch/inference-model.git
%cd /kaggle/working/inference-model
!python scripts/run_pipeline.py --config-name=baseline_v1
```

Launcher сам находит `collection.tsv` внутри `/kaggle/input`, в том числе при
вложенном пути вида
`/kaggle/input/datasets/iskulovdamir/msmarco/collection.tsv`.

Baseline требует чистый Git checkout и использует две CUDA GPU. Результат сохраняется
в:

```text
/kaggle/working/inference-model/outputs/baseline_v1_msmarco_splade_seismic
```

После завершения сохраните **весь** этот каталог как новую версию Kaggle
Dataset. Notebook output без vectors, index, run и manifest не является
воспроизводимым baseline.

> В конце pipeline вычисляет SHA256 больших SPLADE-векторов и SEISMIC-индекса.
> Несколько дополнительных минут после расчёта метрик — ожидаемое поведение.

## Что именно фиксирует baseline

| Компонент | Значение |
|---|---|
| Dataset | MS MARCO Passage, 8 841 823 документа |
| Queries/qrels | Pyserini `msmarco-passage-dev-subset`, 6 980 запросов |
| Model | `naver/splade-cocondenser-selfdistil` |
| Model revision | `31f3eb4b5aae648cde974db3b7b44bf2d095106b` |
| Encoding | 2 × GPU, FP16, batch 32, max length 256 |
| Length bucketing | окно 1 024 документа, dynamic batch-local padding |
| Document vectors | raw SPLADE weights, gzip JSONL shards |
| SEISMIC build | postings 3000, centroid fraction 0.2, summary energy 0.5 |
| SEISMIC search | top-1000, query cut 10, heap factor 0.8 |
| CPU threads | 4 |
| Seed | 42, deterministic PyTorch operations |

Разрешённая Hydra-конфигурация сохраняется рядом с результатами. Поэтому
источником истины для конкретного запуска служат `config.yaml`, `report.json` и
`manifest.json`, а не значения в этой таблице.

## Этапы pipeline

```text
prepare
  -> encode_documents
  -> encode_queries
  -> build_index
  -> search
  -> metrics
```

| Этап | Назначение | Основной результат |
|---|---|---|
| `prepare` | Проверка данных, Git и окружения | dataset/environment fingerprints |
| `encode_documents` | Multi-GPU SPLADE encoding | gzip JSONL shards |
| `encode_queries` | SPLADE encoding dev queries | `queries.dev.jsonl` |
| `build_index` | Построение SEISMIC | `.index.seismic` |
| `search` | Retrieval top-1000 | TREC run |
| `metrics` | Quality и итоговый report | `report.json` |

Этапы атомарны и возобновляемы. Повтор той же команды с тем же
`experiment.name` пропускает только те этапы, у которых совпали конфигурация,
входы и inventory выходных файлов.

Принудительный повтор выбранных этапов:

```bash
python scripts/run_pipeline.py \
  --config-name=baseline_v1 \
  'pipeline.force=[search,metrics]'
```

## Артефакты запуска

```text
outputs/<experiment>/
├── config.yaml                         # полностью разрешённый Hydra config
├── environment.json                    # Git, Python, packages, CPU/GPU/CUDA
├── packages.freeze.txt                 # версии установленных пакетов
├── manifest.json                       # lineage и content hashes
├── report.json                         # quality, speed, sizes and timing
├── vectors/
│   ├── documents/
│   │   ├── part-r00-*.jsonl.gz
│   │   ├── part-r01-*.jsonl.gz
│   │   └── encoding_summary.json
│   └── queries.dev.jsonl
├── seismic/
│   ├── msmarco.index.seismic
│   └── run.dev.trec
├── metrics/per_query.jsonl
├── logs/
│   ├── pipeline.log
│   └── encode.rank-*.log
└── state/*.json                         # resume state каждого этапа
```

`report.json` содержит MRR, nDCG, MAP, Recall, Precision и Success на заданных
cutoffs, времена этапов, docs/s, QPS, latency percentiles и размеры артефактов.

`manifest.json` содержит:

- Git commit и признак dirty checkout;
- hashes dataset, queries, qrels и resolved config;
- SHA256 vectors, index, run, metrics и logs;
- `configuration_id` для конфигурации и `run_id` для конкретных артефактов;
- ссылку и hash родительского manifest для дочерних экспериментов.

## Проверка опубликованного baseline Dataset

Для Dataset
[`iskulovdamir/msmarco-splade-seismic`](https://www.kaggle.com/datasets/iskulovdamir/msmarco-splade-seismic/data)
не нужно повторять encoding или indexing. Подключите только этот Dataset к
Kaggle Notebook; GPU для проверки не требуется. Первая ячейка находит mount
point независимо от того, плоский он или вложенный:

```python
!git clone --depth 1 --branch damir_branch https://github.com/uandrosova07-arch/inference-model.git
%cd /kaggle/working/inference-model

from pathlib import Path

candidates = [
    manifest.parent
    for manifest in Path("/kaggle/input").rglob("manifest.json")
    if (manifest.parent / "seismic/msmarco.index.seismic").is_file()
]
assert len(candidates) == 1, candidates
BASELINE = candidates[0]
print(BASELINE)

!python scripts/verify_kaggle_dataset.py {BASELINE}
```

Структурная проверка должна сообщить `886` document-vector shards и завершиться
строкой `OK`. Kaggle распаковывает каждый загруженный `.jsonl.gz` в каталог с
вложенным `.jsonl.gz.tmp`; launcher поддерживает и исходную плоскую структуру,
и эту структуру mount point.

Проверка входа с этапа `search` запускает только `search -> metrics` поверх
готового 8.6 GB индекса:

```python
!python scripts/run_pipeline.py \
  --config-name=kaggle \
  experiment.name=verify_published_search \
  experiment.kind=verification \
  experiment.parent_manifest={BASELINE}/manifest.json \
  experiment.change_note=verify_published_index_search_and_metrics \
  pipeline.start_at=search \
  paths.index_file={BASELINE}/seismic/msmarco.index.seismic \
  paths.query_vectors={BASELINE}/vectors/queries.dev.jsonl \
  index.threads=4 \
  runtime.require_cuda=false \
  runtime.fingerprint_artifacts=false
```

Отдельно проверьте вход непосредственно с `metrics`, используя опубликованный
TREC run. Повторная установка пакетов уже не нужна:

```python
!python scripts/run_pipeline.py \
  --skip-install \
  --config-name=kaggle \
  experiment.name=verify_published_metrics \
  experiment.kind=verification \
  experiment.parent_manifest={BASELINE}/manifest.json \
  experiment.change_note=verify_published_run_metrics \
  pipeline.start_at=metrics \
  paths.run={BASELINE}/seismic/run.dev.trec \
  runtime.require_cuda=false \
  runtime.fingerprint_artifacts=false
```

Финальная ячейка требует совпадения основных метрик нового search run с
зафиксированным manifest с допуском `1e-6` для различий порядка float/ties между
runtime:

```python
import json
import math
from pathlib import Path

expected = json.loads((BASELINE / "manifest.json").read_text())["effectiveness"]
actual = json.loads(
    Path("outputs/verify_published_search/report.json").read_text()
)["effectiveness"]

for name in ("mrr_at_10", "ndcg_at_10", "map_at_1000", "recall_at_1000"):
    assert math.isclose(actual[name], expected[name], rel_tol=0, abs_tol=1e-6), (
        name,
        actual[name],
        expected[name],
    )
assert actual["queries"] == actual["queries_ranked"] == 6980
print("OK: published index reproduces baseline search and metrics")
```

`runtime.fingerprint_artifacts=false` используется только для быстрой проверки,
чтобы не перечитывать 8.6 GB индекс ради SHA256. Для реального дочернего
эксперимента этот override следует убрать.

## Эксперименты поверх baseline

Правило: один эксперимент — одно осмысленное изменение, новое
`experiment.name`, заполненный `change_note` и ссылка на baseline manifest.

Путь `/kaggle/input/baseline-v1` ниже — пример mount point сохранённого baseline
Dataset. Замените его на фактический путь в Kaggle.

### 1. Search sweep без повторного indexing

```bash
python scripts/run_pipeline.py \
  --config-name=kaggle \
  experiment.name=exp_query_cut_08 \
  experiment.kind=search \
  experiment.parent_manifest=/kaggle/input/baseline-v1/manifest.json \
  experiment.change_note=query_cut_10_to_8 \
  pipeline.start_at=search \
  paths.index_file=/kaggle/input/baseline-v1/seismic/msmarco.index.seismic \
  paths.query_vectors=/kaggle/input/baseline-v1/vectors/queries.dev.jsonl \
  index.threads=4 \
  index.search.query_cut=8
```

Подходит для `query_cut`, `heap_factor`, `first_sorted`, `top_k` и числа search
threads.

### Query-adaptive router: один Kaggle-запуск

Первый learned-router намеренно меняет только `query_cut`: кандидаты `6`, `8`
и baseline `10` используют один и тот же индекс, query vectors,
`heap_factor=0.8`, `top_k=1000`, четыре search thread и выключенный graph.
Дерево глубины 3 получает только дешёвые признаки готового SPLADE-вектора
(число ненулевых компонент, нормы, концентрацию top weights и entropy).

Для каждого запроса oracle выбирает минимальный `query_cut`, который не хуже
`query_cut=10` одновременно по `Recall@1000` и `MRR@10`; фактическое время
каждой фиксированной политики также сохраняется в training report. Модель обучается на
стабильном hash-split `60% train / 20% validation / 20% test`; test не
используется при построении дерева. `leaf_safety_quantile=0.95` делает листья
консервативными для редких сложных запросов.

```bash
python scripts/run_query_router_experiment.py \
  --baseline-root /kaggle/input/datasets/iskulovdamir/msmarco-splade-seismic \
  --experiment-prefix query_router_v1
```

Скрипт последовательно:

1. выполняет search-only кандидаты `query_cut=6` и `8`;
2. переиспользует опубликованный run как политику `query_cut=10`;
3. обучает маленькое ordinal CART-дерево без `sklearn`;
4. сохраняет unbiased test-оценку simulated router;
5. запускает adaptive search, группируя запросы по выбранной политике;
6. выполняет обычную стадию metrics и сохраняет manifest дочернего запуска.

Артефакты модели находятся в
`outputs/query_router_v1_model/`: `training_plan.json`, `router.json`,
`training_report.json`, `training_assignments.jsonl`, `test.qrels.tsv` и
`experiment_summary.json`. Итоговый adaptive run находится в
`outputs/query_router_v1_adaptive/`; его
`metrics/router_assignments.jsonl` содержит решение и признаки для каждого
запроса.

Для чернового iteration можно добавить `--skip-fingerprints`, но финальный
сравниваемый запуск следует делать без этого флага, чтобы manifests содержали
SHA256 всех артефактов. Изменить сетку можно через `--query-cuts 5,7,10`, а
размер дерева — через `--max-depth` и `--min-samples-leaf`.

### 2. Index sweep без повторного SPLADE encoding

```bash
python scripts/run_pipeline.py \
  --config-name=kaggle \
  experiment.name=exp_postings_4000 \
  experiment.kind=index \
  experiment.parent_manifest=/kaggle/input/baseline-v1/manifest.json \
  experiment.change_note=n_postings_3000_to_4000 \
  pipeline.start_at=build_index \
  paths.document_vectors=/kaggle/input/baseline-v1/vectors/documents \
  paths.query_vectors=/kaggle/input/baseline-v1/vectors/queries.dev.jsonl \
  index.threads=4 \
  index.n_postings=4000
```

Подходит для `n_postings`, `centroid_fraction`, `summary_energy`,
`max_fraction`, `doc_cut` и вариантов clustering.

### 3. Encoder experiment с полным downstream evaluation

Пример: увеличить окно length bucketing с 1 024 до 2 048 документов, сохранив
batch size 32.

```bash
python scripts/run_pipeline.py \
  --config-name=kaggle \
  experiment.name=exp_bucket_2048 \
  experiment.kind=encoder \
  experiment.parent_manifest=/kaggle/input/baseline-v1/manifest.json \
  experiment.change_note=bucket_window_1024_to_2048 \
  model.devices=2 \
  model.precision=fp16 \
  model.bucket_multiplier=64 \
  index.threads=4
```

Изменения `batch_size`, `bucket_multiplier`, `max_length`, `threshold`,
`max_terms`, precision или модели требуют повторного document/query encoding и
всех downstream-этапов.

### 4. Только пересчёт метрик

```bash
python scripts/run_pipeline.py \
  --config-name=kaggle \
  experiment.name=metrics_only \
  experiment.kind=metrics \
  experiment.parent_manifest=/kaggle/input/baseline-v1/manifest.json \
  experiment.change_note=independent_metric_verification \
  pipeline.start_at=metrics \
  paths.run=/kaggle/input/baseline-v1/seismic/run.dev.trec
```

Для собственных TSV дополнительно задайте:

```bash
dataset.query_source=file \
dataset.queries=/kaggle/input/my-data/queries.tsv \
dataset.qrels_source=file \
dataset.qrels=/kaggle/input/my-data/qrels.tsv
```

## Работа с локальной моделью и offline Kaggle

Модель можно подключить отдельным Kaggle Dataset. Для полного baseline:

```bash
python scripts/run_pipeline.py \
  --config-name=baseline_v1 \
  model.local_path=/kaggle/input/splade-model/splade-cocondenser-selfdistil \
  model.local_files_only=true
```

Если пакеты подготовлены в wheelhouse:

```bash
python scripts/run_pipeline.py \
  --config-name=baseline_v1 \
  --wheelhouse=/kaggle/input/retrieval-wheels \
  model.local_path=/kaggle/input/splade-model/splade-cocondenser-selfdistil \
  model.local_files_only=true
```

### Запуск собственного checkpoint `splade-baseline-encoder`

Backend `local_splade` загружает каталог с `config.json`, tokenizer files и
`model.pt` через собственный `Splade`-совместимый encoder и передаёт его
векторы в тот же pipeline: SEISMIC, поиск и метрики. Подключите к Kaggle уже
существующий MS MARCO и Dataset с файлами модели, затем выполните из checkout
ветки с этой интеграцией:

```bash
python scripts/run_pipeline.py \
  --config-name=kaggle \
  experiment.name=my_splade_msmarco \
  model.backend=local_splade \
  model.local_path=/kaggle/input/<model-dataset>/model \
  model.local_files_only=true \
  model.max_length=256 \
  model.devices=2
```

Если каталог модели включён в checkout, укажите относительный путь
`model.local_path=splade-baseline-encoder/model`. Launcher автоматически
находит `collection.tsv` в подключённом MS MARCO. Полный отчёт, включая
MRR@10, nDCG@10, MAP@1000, Recall@1000, Precision@10 и Success@10, сохраняется
в `outputs/my_splade_msmarco/report.json`.

## Локальная разработка

Требуется Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Smoke pipeline использует до 1 000 документов и 100 запросов:

```bash
python pipeline.py --config-name=smoke
```

Для него нужны локальные файлы:

```text
input/datasets/msmarco/collection.tsv
input/datasets/msmarco/queries.dev.tsv
input/datasets/msmarco/qrels.dev.tsv
```

Проверить итоговую Hydra-конфигурацию без запуска pipeline:

```bash
python pipeline.py --config-name=baseline_v1 --cfg job
```

## Структура проекта

```text
.
├── pipeline.py                    # Hydra entrypoint
├── scripts/run_pipeline.py        # Kaggle bootstrap and launcher
├── src/
│   ├── configs/                   # baseline, dataset, model, index, metrics
│   ├── datasets/                  # MS MARCO readers
│   ├── model/                     # SPLADE encoding and multi-GPU sharding
│   ├── index/                     # SEISMIC build and search
│   ├── metrics/                   # retrieval metrics
│   ├── pipeline/                  # orchestration, resume and manifests
│   └── utils/                     # IO, logging and reproducibility
└── tests/
```

## Legacy и специализированные сценарии

Для новых baseline и экспериментов используйте `scripts/run_pipeline.py`.
Остальные entrypoints сохранены для совместимости и отдельных сравнений:

| Скрипт | Назначение |
|---|---|
| `scripts/index_baseline.py` | standalone SPLADE document encoding |
| `scripts/run_kaggle_pipeline.py` | старый SPLADE + Lucene workflow |
| `scripts/run_seismic_pipeline.py` | SEISMIC поверх ранее сохранённых vectors |
| `scripts/run_precomputed_benchmark.py` | Lucene vs SEISMIC на готовых vectors |
| `scripts/evaluate_run.py` | независимый пересчёт retrieval metrics |

Готовые SPLADE-векторы для отдельного benchmark доступны в
[`tuskanny/seismic-msmarco-splade`](https://huggingface.co/datasets/tuskanny/seismic-msmarco-splade/tree/main).

## Ссылки

- [MS MARCO Passage Ranking](https://microsoft.github.io/msmarco/Datasets.html)
- [SPLADE CoCondenser SelfDistil](https://huggingface.co/naver/splade-cocondenser-selfdistil)
- [SPLADE](https://arxiv.org/abs/2107.05720)
- [SPLADE v2](https://arxiv.org/abs/2109.10086)
- [From Distillation to Hard Negative Sampling](https://arxiv.org/abs/2205.04733)
- [SEISMIC](https://arxiv.org/abs/2404.18812)
- [Understanding Wacky Weights](https://arxiv.org/abs/2401.03366)
