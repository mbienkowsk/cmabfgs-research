from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_DIR_STR = str(CONFIG_DIR)

DATA_DIR = ROOT_DIR / "data"
CMAES_DATA_DIR = DATA_DIR / "cmaes"
CMAES_C_METRICS_DIR = CMAES_DATA_DIR / "c_metrics"
