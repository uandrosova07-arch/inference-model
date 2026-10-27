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
