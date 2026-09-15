"""Central configuration: paths, column names and constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_DOCS = ROOT / "data" / "docs"
MODELS = ROOT / "models"
CHROMA_DIR = ROOT / "chroma_db"
MLFLOW_DB = ROOT / "mlflow.db"

AI4I_CSV = DATA_RAW / "ai4i2020.csv"
NEU_RAW = DATA_RAW / "NEU-CLS"

CLEAN_CSV = DATA_PROCESSED / "ai4i_clean.csv"
SEGMENTS_CSV = DATA_PROCESSED / "machine_segments.csv"
NEU_IMAGES = DATA_PROCESSED / "neu_images"

for _d in (DATA_PROCESSED, DATA_DOCS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- module 1
TARGET = "machine_failure"

# Leakage columns. `machine_failure` is the OR of these five flags, so leaving
# them in as features makes every model score ~100% and the comparison
# meaningless. They are dropped during preparation.
LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "type_code",        # H=0, M=1, L=2 (ordinal: higher = lower quality)
    "temp_diff_k",      # process - air   -> drives heat dissipation failure
    "power_w",          # torque * angular velocity -> drives power failure
    "wear_torque",      # tool wear * torque -> drives overstrain failure
]

FEATURE_LABELS = {
    "air_temp_k": "Air temperature (K)",
    "process_temp_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
    "type_code": "Product quality variant",
    "temp_diff_k": "Temperature difference (K)",
    "power_w": "Mechanical power (W)",
    "wear_torque": "Wear x torque",
}

TYPE_MAP = {"H": 0, "M": 1, "L": 2}
TYPE_NAMES = {0: "H (high quality)", 1: "M (medium)", 2: "L (low quality)"}

# ---------------------------------------------------------------- module 2
N_CLUSTERS = 4
CLUSTER_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

# ---------------------------------------------------------------- module 3
DEFECT_CLASSES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
DEFECT_NAMES = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted Surface",
    "RS": "Rolled-in Scale",
    "Sc": "Scratches",
}
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
VISION_MODEL = MODELS / "resnet50_defects.keras"
VISION_LABELS = MODELS / "defect_classes.json"

# ---------------------------------------------------------------- module 4
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | google
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------- database
DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "name": os.getenv("DB_NAME", "smart_manufacturing"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

MLFLOW_URI = f"sqlite:///{MLFLOW_DB}"
MLFLOW_EXPERIMENT = "predictive_maintenance"

RANDOM_STATE = 42
