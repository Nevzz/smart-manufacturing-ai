"""Smart Manufacturing AI -- Streamlit entry point.

Run from the project root:  streamlit run app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import config, database

st.set_page_config(
    page_title="Smart Manufacturing AI",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Smart Manufacturing AI")
st.caption("Predictive Maintenance & Visual Quality Inspection")

st.markdown(
    """
This application brings four analytics modules together on one shop-floor
dataset. Use the sidebar to move between them.

| Module | Question it answers |
|---|---|
| **Predictive Maintenance** | Which machines are about to fail, and what drives failure? |
| **Machine Segmentation** | What distinct operating regimes exist, and which is riskiest? |
| **Defect Detection** | What kind of surface defect is in this image? |
| **Maintenance Assistant** | What should a technician actually do about it? |
"""
)

st.divider()

# ------------------------------------------------------------------ status
st.subheader("Pipeline status")

checks = [
    ("Sensor data prepared", config.CLEAN_CSV.exists(), "python src/prepare_data.py"),
    ("Defect images prepared", config.NEU_IMAGES.exists(), "python src/prepare_data.py"),
    ("Maintenance models trained", (config.MODELS / "best_model.pkl").exists(),
     "python src/train_maintenance.py"),
    ("Segmentation trained", config.SEGMENTS_CSV.exists(),
     "python src/train_segmentation.py"),
    ("Defect classifier trained", config.VISION_MODEL.exists(),
     "python src/train_vision.py"),
    ("Knowledge base built", config.CHROMA_DIR.exists(), "python src/build_rag.py"),
]

cols = st.columns(3)
for i, (label, ok, cmd) in enumerate(checks):
    with cols[i % 3]:
        if ok:
            st.success(f"✓ {label}")
        else:
            st.warning(f"○ {label}\n\n`{cmd}`")

db_ok = database.available()
st.caption(
    f"PostgreSQL: {'connected' if db_ok else 'not connected — reading from CSV instead'}"
)

# ------------------------------------------------------------------ summary
if config.CLEAN_CSV.exists():
    st.divider()
    st.subheader("Dataset at a glance")

    df = pd.read_csv(config.CLEAN_CSV)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machine readings", f"{len(df):,}")
    c2.metric("Failures", f"{int(df[config.TARGET].sum()):,}")
    c3.metric("Failure rate", f"{df[config.TARGET].mean():.2%}")
    c4.metric("Defect images", "1,800")

    st.info(
        "Only 3.4% of readings are failures. Accuracy alone is misleading here — "
        "a model that always predicts 'no failure' scores 96.6% and catches nothing. "
        "The comparison in the Predictive Maintenance module is judged on recall, "
        "F1 and ROC-AUC instead."
    )
