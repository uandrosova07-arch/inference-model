# Sparse Retrieval

Улучшение инференса моделей разреженного поиска / улучшение процесса обучения моделей семейства SPLADE

**Данные**
- [MS Marco (Passage Ranking)](https://microsoft.github.io/msmarco/Datasets.html)

**Модели**
- [splade-cocondenser-selfdistil](https://huggingface.co/naver/splade-cocondenser-selfdistil)

**Статьи**
- [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086)
- [Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations](https://arxiv.org/abs/2404.18812)
- [Efficient Sketching and Nearest Neighbor Search Algorithms for Sparse Vector Sets](https://arxiv.org/abs/2509.24815)

## Полный прогон на Kaggle: 2 × T4

Настройки Kaggle:

- Accelerator: `GPU T4 x2`;
- подключённый датасет: `iskulovdamir/msmarco`;
- путь к данным: `/kaggle/input/msmarco`;
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
