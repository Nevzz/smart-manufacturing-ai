"""Step 1 of the pipeline: clean the AI4I CSV and reorganise the NEU images.

Run:  python src/prepare_data.py
"""

from __future__ import annotations

import random
import shutil

import numpy as np
import pandas as pd

try:
    from src import config, database
except ImportError:
    import config
    import database


RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}


def prepare_ai4i() -> pd.DataFrame:
    print("\n[1/2] Preparing AI4I 2020 sensor data")
    df = pd.read_csv(config.AI4I_CSV)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    df = df.rename(columns=RENAME)

    print(f"  loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  missing values: {int(df.isna().sum().sum())}")

    # Keep the failure-mode flags for descriptive analysis, but record clearly
    # that they are NOT features.
    failure_modes = df[config.LEAKAGE_COLS].copy()

    # ---- feature engineering -------------------------------------------
    df["type_code"] = df["type"].map(config.TYPE_MAP)
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    # torque (Nm) x angular velocity (rad/s) = mechanical power (W)
    df["power_w"] = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60
    df["wear_torque"] = df["tool_wear_min"] * df["torque_nm"]

    # dominant failure mode, for reporting only
    def mode_label(row):
        for flag in config.LEAKAGE_COLS:
            if row[flag] == 1:
                return flag
        return "None"

    df["failure_mode"] = failure_modes.apply(mode_label, axis=1)

    keep = ["udi", "product_id", "type"] + config.FEATURES + [config.TARGET, "failure_mode"]
    clean = df[keep].copy()

    rate = clean[config.TARGET].mean()
    print(f"  failures: {int(clean[config.TARGET].sum())} ({rate:.2%}) -- imbalanced")
    print(f"  dropped leakage columns: {', '.join(config.LEAKAGE_COLS)}")

    clean.to_csv(config.CLEAN_CSV, index=False)
    print(f"  saved -> {config.CLEAN_CSV.relative_to(config.ROOT)}")

    database.save_table(clean, "sensor_readings")
    return clean


def prepare_neu(train_frac: float = 0.8) -> None:
    print("\n[2/2] Preparing NEU surface-defect images")

    if not config.NEU_RAW.exists():
        print(f"  !! {config.NEU_RAW} not found -- extract NEU-CLS.rar first")
        return

    if config.NEU_IMAGES.exists():
        shutil.rmtree(config.NEU_IMAGES)

    random.seed(config.RANDOM_STATE)
    total = 0
    for cls in config.DEFECT_CLASSES:
        files = sorted(config.NEU_RAW.glob(f"{cls}_*.bmp"))
        random.shuffle(files)
        split = int(train_frac * len(files))
        for subset, items in (("train", files[:split]), ("val", files[split:])):
            out = config.NEU_IMAGES / subset / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy(f, out / f.name)
        total += len(files)
        print(f"  {cls} ({config.DEFECT_NAMES[cls]}): {split} train / {len(files) - split} val")

    print(f"  {total} images -> {config.NEU_IMAGES.relative_to(config.ROOT)}")


def main() -> None:
    print("=" * 60)
    print("SMART MANUFACTURING AI -- data preparation")
    print("=" * 60)
    prepare_ai4i()
    prepare_neu()
    print("\nDone. Next: python src/train_maintenance.py\n")


if __name__ == "__main__":
    main()
