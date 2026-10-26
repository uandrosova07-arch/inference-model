import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
LOCAL_DATASETS_DIR = INPUT_DIR / "datasets"
KAGGLE_DATASETS_DIR = Path("/kaggle/input/msmarco")
MODELS_DIR = INPUT_DIR / "models"

BASELINE_MODEL_PROVIDER = "naver"
BASELINE_MODEL_NAME = "splade-cocondenser-selfdistil"
BASELINE_MODEL_ID = f"{BASELINE_MODEL_PROVIDER}/{BASELINE_MODEL_NAME}"
BASELINE_MODEL_DIR = MODELS_DIR / BASELINE_MODEL_NAME


def msmarco_dir() -> Path:
    """Resolve either the Kaggle mount, an explicit env path, or local data."""
    configured = os.environ.get("MSMARCO_DATA_DIR")
    if configured:
        return Path(configured)
    if (KAGGLE_DATASETS_DIR / "collection.tsv").exists():
        return KAGGLE_DATASETS_DIR
    return LOCAL_DATASETS_DIR / "msmarco"


def baseline_model_path() -> str:
    configured = os.environ.get("BASELINE_MODEL_PATH")
    if configured:
        return configured
    if BASELINE_MODEL_DIR.exists():
        return str(BASELINE_MODEL_DIR)
    return BASELINE_MODEL_ID
