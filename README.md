# Sparse Retrieval Inference Boost

Улучшение инференса моделей разреженного поиска / улучшение процесса обучения моделей семейства SPLADE

**Данные**
- [MS Marco (Passage Ranking)](https://microsoft.github.io/msmarco/Datasets.html)

**Модели**
- [splade-cocondenser-selfdistil](https://huggingface.co/naver/splade-cocondenser-selfdistil)

**Статьи**
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.google.com/url?source=gmail&sa=E&q=https://arxiv.org/abs/1810.04805)
- [Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations](https://www.google.com/url?source=gmail&sa=E&q=https://arxiv.org/abs/2404.18812)
- [Membership Inference Attacks Against Machine Learning Models](https://www.google.com/url?source=gmail&sa=E&q=https://arxiv.org/abs/1610.05820)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://www.google.com/url?source=gmail&sa=E&q=https://arxiv.org/abs/2107.05720)
- [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://www.google.com/url?source=gmail&sa=E&q=https://arxiv.org/abs/2109.10086)
- [Understanding Wacky Weights: A Dissection of SPLADE's Learned Term Importance](https://www.google.com/url?source=gmail&sa=E&q=https://arxiv.org/abs/2401.03366)

## Единый экспериментальный pipeline

Основной сценарий теперь запускается одной командой и состоит из независимых,
возобновляемых этапов:

```text
collection.tsv
    -> prepare
    -> encode_documents (SPLADE, 1..N GPU, resumable JSONL shards)
    -> encode_queries   (SPLADE)
    -> build_index      (SEISMIC)
    -> search           (SEISMIC)
    -> metrics          (quality + speed + artifact sizes)
```

Структура повторяет подход
[`Blinorot/pytorch_project_template`](https://github.com/Blinorot/pytorch_project_template):
код разделён по `datasets`, `model`, `index`, `metrics`, orchestration находится
в `src/pipeline`, а все параметры эксперимента собраны в Hydra-конфигах.

```text
.
├── pipeline.py                 # единая точка входа
├── src
│   ├── configs                 # dataset/model/index/metrics + Kaggle/smoke
│   ├── datasets                # чтение MS MARCO и qrels
│   ├── model                   # SPLADE и multi-GPU sharding
│   ├── index                   # build/search SEISMIC
│   ├── metrics                 # MRR, nDCG, MAP, Recall, Precision, Success
│   ├── pipeline                # resume, диапазоны этапов, state
│   └── utils
├── scripts/run_pipeline.py     # Kaggle bootstrap + запуск
└── tests
```

### Полный прогон на Kaggle

Настройки notebook: `Accelerator = GPU T4 x2`, `Internet = On`, подключённый
датасет [`iskulovdamir/msmarco`](https://www.kaggle.com/datasets/iskulovdamir/msmarco).
Launcher автоматически находит `collection.tsv` в подключённых Kaggle Dataset,
включая вложенный путь вида
`/kaggle/input/datasets/iskulovdamir/msmarco/collection.tsv`. Конфиг включает
gzip для SPLADE-шардов и сохраняет результаты в `/kaggle/working`. Если
подключено несколько коллекций, путь можно задать явно через
`dataset.documents=/path/to/collection.tsv`.

Полный notebook-блок с клонированием:

```bash
!git clone --depth 1 --branch damir_branch https://github.com/uandrosova07-arch/inference-model.git
%cd /kaggle/working/inference-model
!python scripts/run_pipeline.py experiment.name=msmarco_splade_seismic
```

Если репозиторий уже находится в текущей папке, нужна только последняя строка.
Launcher проверяет закреплённые зависимости, модель загружается один раз с
Hugging Face, а число GPU определяется автоматически. Повтор той же команды в
том же session пропускает завершённые этапы и продолжает недописанные
атомарные шарды.

Если модель подключена отдельным Kaggle Dataset и Internet нужен только для
установки пакетов:

```bash
python scripts/run_pipeline.py \
  experiment.name=msmarco_splade_seismic \
  model.local_path=/kaggle/input/splade-model/splade-cocondenser-selfdistil
```

При отключённом Internet зависимости можно положить в wheelhouse-датасет:

```bash
python scripts/run_pipeline.py \
  --wheelhouse /kaggle/input/retrieval-wheels \
  experiment.name=offline_run
```

Все результаты пишутся в
`/kaggle/working/inference-model/outputs/<experiment.name>`. После полного
запуска сохраните notebook version с output, иначе новый Kaggle session не
увидит `/kaggle/working`. SEISMIC получает gzip-шарды потоково через FIFO в два
прохода: дополнительная полноразмерная копия векторов на диске не создаётся.

### Запуск с любого этапа

Диапазон задаётся включительно через `pipeline.start_at` и `pipeline.end_at`.
Допустимые этапы: `prepare`, `encode_documents`, `encode_queries`,
`build_index`, `search`, `metrics`.

Только получить SPLADE-векторы:

```bash
python scripts/run_pipeline.py \
  experiment.name=vectors_v1 \
  pipeline.end_at=encode_queries
```

Построить SEISMIC из добавленных готовых векторов (`.jsonl`, каталог шардов
или `.tar.gz`) и продолжить до метрик:

```bash
python scripts/run_pipeline.py \
  experiment.name=from_vectors \
  pipeline.start_at=build_index \
  paths.document_vectors=/kaggle/input/splade-vectors/documents \
  paths.query_vectors=/kaggle/input/splade-vectors/queries.jsonl \
  dataset.expected_documents=8841823
```

Начать с готового индекса:

```bash
python scripts/run_pipeline.py \
  experiment.name=from_index \
  pipeline.start_at=search \
  paths.index_file=/kaggle/input/seismic-index/msmarco.index.seismic \
  paths.query_vectors=/kaggle/input/splade-vectors/queries.jsonl
```

Пересчитать только метрики для готового TREC/MS MARCO run:

```bash
python scripts/run_pipeline.py \
  experiment.name=metrics_only \
  pipeline.start_at=metrics \
  paths.run=/kaggle/input/my-run/run.trec
```

Для собственного query/qrels TSV вместо ресурсов Pyserini:

```bash
python scripts/run_pipeline.py \
  pipeline.start_at=metrics \
  paths.run=/kaggle/input/my-run/run.trec \
  dataset.query_source=file \
  dataset.queries=/kaggle/input/my-data/queries.tsv \
  dataset.qrels_source=file \
  dataset.qrels=/kaggle/input/my-data/qrels.tsv
```

`pipeline.force=[search,metrics]` принудительно повторяет выбранные этапы.
Для отдельного набора гиперпараметров рекомендуется менять
`experiment.name`: так результаты разных запусков не смешиваются.

### Метрики и артефакты

`outputs/<experiment>/report.json` объединяет:

- MRR, nDCG, MAP, Recall, Precision и Success на всех cutoffs из конфига;
- per-stage wall time, docs/s, QPS и batch-amortized p50/p95/p99;
- размеры SPLADE-векторов, SEISMIC-индекса и run-файла;
- параметры, environment/package/GPU snapshot и пути конкретного запуска.

Per-query значения сохраняются в `metrics/per_query.jsonl`, разрешённый Hydra
конфиг — в `config.yaml`, состояние каждого этапа — в `state/`, основной лог —
в `logs/pipeline.log`, а GPU worker logs — в `logs/encode.rank-XX.log`.
Одновременно те же сообщения видны в notebook output. Параметры SEISMIC
меняются обычными overrides, например:

```bash
python scripts/run_pipeline.py \
  experiment.name=seismic_sweep_01 \
  pipeline.start_at=build_index \
  paths.document_vectors=/kaggle/input/splade-vectors/documents.jsonl \
  paths.query_vectors=/kaggle/input/splade-vectors/queries.jsonl \
  index.n_postings=4000 \
  index.centroid_fraction=0.1 \
  index.search.query_cut=8 \
  index.search.heap_factor=0.7
```

Для словаря больше 65 535 компонент используйте
`index.variant=large_vocab`. Обычный BERT/SPLADE vocab помещается в
`index.variant=standard`.

Локальная быстрая проверка (1 000 документов / 100 запросов):

```bash
pip install -r requirements.txt
python pipeline.py --config-name=smoke
pytest -q
```

## Совместимые специализированные сценарии

Старые entrypoints оставлены для воспроизводимости уже начатых запусков и
Lucene-сравнения. Новый экспериментальный код лучше запускать через
`scripts/run_pipeline.py`.

### Полный прогон на Kaggle: 2 × T4

Настройки Kaggle:

- Accelerator: `GPU T4 x2`;
- подключённый датасет: `iskulovdamir/msmarco`;
- путь к данным определяется автоматически по `collection.tsv`;
- Internet нужен для `git clone`, установки Pyserini и однократной загрузки модели.

Артефакты создаются в `/kaggle/working`; после завершения нужно сохранить
версию запуска в Kaggle, иначе новый сеанс не увидит готовые шарды. Если модель
подключена отдельным Kaggle Dataset, передайте её путь через `--model` и Internet
для её загрузки не понадобится.

Основной датасет скрипт не скачивает. Одна команда запускает два независимых
процесса без сетевого rendezvous; каждый процесс закреплён за своей T4 и
обрабатывает свою половину строк коллекции. По умолчанию используется FP16.

```bash
PYTHONPATH=. python scripts/run_kaggle_pipeline.py \
  --data-dir /kaggle/input/msmarco \
  --output-dir /kaggle/working/inference-model/outputs/baseline \
  --gpus 2 \
  --batch-size 32 \
  --bucket-multiplier 32 \
  --max-length 256
```

Pipeline выполняет:

1. параллельное кодирование на двух T4;
2. построение Lucene impact index;
3. кодирование стандартных 6 980 dev-запросов из Pyserini;
4. retrieval top-1000;
5. MRR@10 и Recall@1000.

Стандартные 6 980 dev-запросов и qrels берутся непосредственно из Pyserini,
поэтому дополнительные локальные файлы для оценки не создаются.

При OOM уменьшите `--batch-size` до 16. Повторный запуск этапа `encode`
продолжает готовые атомарные шарды; отдельный этап запускается через
`--stage encode|index|queries|search|evaluate`.

Итоговые метрики:

- `outputs/baseline/encoding_summary.json` — общая скорость двух GPU и padding;
- `outputs/baseline/pipeline_metrics.json` — время и размер артефактов;
- `outputs/baseline/retrieval_metrics.json` — MRR@10 и Recall@1000.

## SEISMIC поверх сохранённых SPLADE-векторов

SEISMIC строится отдельным CPU-этапом, поэтому повторно кодировать коллекцию на
GPU не требуется. Скрипт потоково объединяет JSONL-шарды во временный файл в
`/kaggle/temp`, проверяет полноту коллекции, строит индекс, удаляет временный
файл и измеряет MRR@10, Recall@1000 и скорость поиска.

```bash
PYTHONPATH=. python scripts/run_seismic_pipeline.py \
  --vectors-dir /kaggle/working/inference-model/outputs/baseline/vectors/world-size-2 \
  --queries /kaggle/working/inference-model/outputs/baseline/queries.dev.splade.tsv \
  --output-dir /kaggle/working/inference-model/outputs/baseline \
  --threads 4
```

Параметры по умолчанию соответствуют высокоточному примеру авторов SEISMIC
для MS MARCO SPLADE Co-Condenser: `n_postings=3000`,
`centroid_fraction=0.2`, `summary_energy=0.5`, `max_fraction=6`, а для поиска —
`query_cut=10`, `heap_factor=0.8`, `first_sorted=false`.

Артефакты:

- `outputs/baseline/seismic/msmarco.index.seismic` — сериализованный индекс;
- `outputs/baseline/seismic/run.dev.tsv` — top-1000 выдача;
- `outputs/baseline/seismic/retrieval_metrics.json` — MRR@10 и Recall@1000;
- `outputs/baseline/seismic/pipeline_metrics.json` — время, QPS и размеры.

## Benchmark на готовых векторах без повторного GPU-кодирования

Официальные готовые представления MS MARCO Passage для SPLADE Co-Condenser
опубликованы в
[`tuskanny/seismic-msmarco-splade`](https://huggingface.co/datasets/tuskanny/seismic-msmarco-splade/tree/main):

- `documents.tar.gz`;
- `queries.tar.gz`.

Архивы можно подключить как Kaggle Dataset. Распаковывать 33 GiB JSONL не
нужно: Lucene индексируется непосредственно из потока архива, а SEISMIC читает
тот же архив через FIFO в два прохода. Команда измеряет четыре отдельных этапа:

1. построение Lucene impact index;
2. поиск Lucene;
3. построение SEISMIC index;
4. поиск SEISMIC.

```bash
PYTHONPATH=. python scripts/run_precomputed_benchmark.py \
  --documents /kaggle/input/msmarco-splade-vectors/documents.tar.gz \
  --queries /kaggle/input/msmarco-splade-vectors/queries.tar.gz \
  --output-dir /kaggle/working/inference-model/outputs/precomputed_benchmark \
  --threads 4
```

Все времена, размеры, QPS, MRR@10, Recall@1000 и относительное ускорение
сохраняются в `outputs/precomputed_benchmark/benchmark_metrics.json`.
