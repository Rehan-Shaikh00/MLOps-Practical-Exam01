"""Feature engineering and offline Parquet feature-store materialization.

Reads ``data/wine_cleaned.csv``, engineers domain ratio/interaction
features, scales all model features with ``StandardScaler``, persists the
scaler artifact, materializes the feature store as Parquet, and writes
``feature_store/registry.json`` describing the feature group.

Run from the repository root with::
    python -m src.feature_engineering
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
INPUT_PATH: Path = PROJECT_ROOT / "data" / "wine_cleaned.csv"
FEATURE_STORE_DIR: Path = PROJECT_ROOT / "feature_store"
SCALER_PATH: Path = FEATURE_STORE_DIR / "wine_scaler.pkl"
STORE_PATH: Path = FEATURE_STORE_DIR / "wine_features.parquet"
REGISTRY_PATH: Path = FEATURE_STORE_DIR / "registry.json"

ENTITY_KEY: str = "wine_id"
TIMESTAMP_KEY: str = "event_timestamp"
TARGET: str = "target"
FEATURE_GROUP: str = "wine_classification_features"
VERSION: str = "v1.0"


def run_feature_engineering() -> dict[str, Any]:
    """Engineer features, scale them, and materialize the Parquet store.

    Returns:
        The feature-store registry dictionary written to ``registry.json``.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {INPUT_PATH}. "
            "Run 'python -m src.preprocessing' first."
        )

    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = pd.read_csv(INPUT_PATH)
    logger.info("Loaded cleaned dataset with shape %s", df.shape)

    # Domain ratio / interaction features (epsilon guards against div-by-zero).
    df["alcohol_to_malic_ratio"] = df["alcohol"] / (df["malic_acid"] + 1e-5)
    df["total_phenols_flavonoids"] = df["total_phenols"] + df["flavanoids"]
    df["color_hue_interaction"] = df["color_intensity"] * df["hue"]

    meta_cols = [ENTITY_KEY, TIMESTAMP_KEY, TARGET]
    feature_cols: list[str] = [c for c in df.columns if c not in meta_cols]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    joblib.dump(scaler, SCALER_PATH)
    logger.info("Persisted fitted scaler to %s", SCALER_PATH)

    df.to_parquet(STORE_PATH, index=False)
    logger.info("Materialized %d features to %s", len(feature_cols), STORE_PATH)

    registry: dict[str, Any] = {
        "feature_group": FEATURE_GROUP,
        "version": VERSION,
        "entity_key": ENTITY_KEY,
        "timestamp_key": TIMESTAMP_KEY,
        "target": TARGET,
        "features": feature_cols,
        "storage_path": str(STORE_PATH.relative_to(PROJECT_ROOT)),
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=4), encoding="utf-8")
    logger.info("Wrote feature-store registry to %s", REGISTRY_PATH)
    return registry


def main() -> None:
    """CLI entry point."""
    run_feature_engineering()


if __name__ == "__main__":
    main()