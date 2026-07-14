from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PHASE2_DIR = PROJECT_ROOT / "phase2"
RX_LEVEL = DATA_DIR / "rx_level.parquet"

N_LEVELS = 4  # Default quartile discrete states 0..3

# Cardinality per node (Peak0 is binary, others are quartiles)
CARDINALITY: dict[str, int] = {
    "Peak0": 2,
    "Load0": 4,
    "N0": 4,
    "M0": 4,
    "W0": 4,
    "D0": 4,
    "V0": 4,
    "Y0": 4,
}

# Peak hours: morning 9-11, afternoon 14-16
PEAK_HOURS = ((9, 11), (14, 16))

# Gibbs defaults
MCMC_DRAWS = 2000
MCMC_BURN = 500
DIRICHLET_ALPHA = 1.0  # Uniform prior
