"""Data ingestion and cleaning for the Wine classification pipeline.
Loads the scikit-learn Wine dataset, sanitizes column names, removes
duplicates and nulls, attaches an entity key (``wine_id``) and a UTC
event timestamp, and persists the cleaned frame to ``data/``.
Run from the repository root with::
 python -m src.preprocessing
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_wine
logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_PATH: Path = DATA_DIR / "wine_cleaned.csv"
ENTITY_KEY: str = "wine_id"
TIMESTAMP_KEY: str = "event_timestamp"
def run_preprocessing() -> Path:
 """Load, clean, and persist the Wine dataset.
 Returns:
 Path to the cleaned CSV written under ``data/``.
 """
 DATA_DIR.mkdir(parents=True, exist_ok=True)
 wine = load_wine(as_frame=True)
 df: pd.DataFrame = wine.frame.copy()
 logger.info("Loaded raw Wine dataset with shape %s", df.shape)
 # Sanitize column names: "od280/od315 of diluted wines" -> "od280_od315_of_diluted_wines"
 df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]
 n_before = len(df)
 df = df.drop_duplicates().dropna().reset_index(drop=True)
 logger.info("Dropped %d duplicate/null rows (kept %d)", n_before - len(df), len(df))
 df.insert(0, ENTITY_KEY, range(1, len(df) + 1))
 df[TIMESTAMP_KEY] = pd.Timestamp.now(tz="UTC")
 df.to_csv(OUTPUT_PATH, index=False)
 logger.info("Saved cleaned dataset to %s | Shape: %s", OUTPUT_PATH, df.shape)
 return OUTPUT_PATH
def main() -> None:
 """CLI entry point."""
 run_preprocessing()
if __name__ == "__main__":
 main()
