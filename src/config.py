"""Central configuration. Import values from here rather than hard-coding them."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
RESULTS_LOG_PATH = REPORTS_PATH / "results_log.csv"

# Chronological split
# Re-check these against the calendar before training/evaluating —
# see .github/copilot-instructions.md #10 for why this can go stale.
TRAIN_START_SEASON = "2010-11"
TRAIN_END_SEASON = "2023-24"
VALIDATION_START_SEASON = "2024-25"
VALIDATION_END_SEASON = "2025-26"
TEST_SEASON = "2026-27"

# Reproducibility
RANDOM_STATE = 42

# Feature engineering
ROLLING_WINDOWS = [5, 10]
