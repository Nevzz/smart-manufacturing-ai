import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")
st.title("🔧 Predictive Maintenance")
st.caption("Logistic Regression vs Decision Tree vs Random Forest")

COMPARISON = config.DATA_PROCESSED / "model_comparison.csv"
IMPORTANCE = config.DATA_PROCESSED / "feature_importance.csv"
ROC = config.DATA_PROCESSED / "roc_curves.json"

if not COMPARISON.exists():
    st.warning("Models not trained yet. Run `python src/train_maintenance.py` first.")
    st.stop()


@st.cache_data
def load_results():
    res = pd.read_csv(COMPARISON)
    imp = pd.read_csv(IMPORTANCE)
    roc = json.loads(ROC.read_text())
    return res, imp, roc


@st.cache_resource
def load_model():
    meta = json.loads((config.MODELS / "best_model.json").read_text())
    return joblib.load(config.MODELS / "best_model.pkl"), meta


res, imp, roc = load_results()
model, meta = load_model()

tab1, tab2, tab3 = st.tabs(["Model comparison", "What drives failure", "Predict"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.subheader("Performance on the held-out test set (2,000 readings)")

    show = res.copy()
    show.columns = [c.replace("_", "-").title() for c in show.columns]
    st.dataframe(
        show.style.format({c: "{:.3f}" for c in show.columns if c != "Model"})
        .background_gradient(cmap="Greens", subset=["F1", "Roc-Auc"]),
        width="stretch",
        hide_index=True,
    )

    long = res.melt(id_vars="model", var_name="metric", value_name="score")
    fig = px.bar(
        long, x="metric", y="score", color="model", barmode="group",
        title="Metric comparison", height=400,
    )
    fig.add_hline(
        y=0.966, line_dash="dot", line_color="grey",
        annotation_text="accuracy of a 'never fails' baseline",
    )
    fig.update_layout(yaxis_range=[0, 1], xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    roc_fig = go.Figure()
    for name, d in roc.items():
        roc_fig.add_trace(
            go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines",
                       name=f"{name} (AUC {d['auc']:.3f})")
        )
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                   line=dict(dash="dash", color="grey"))
    )
    roc_fig.update_layout(
        title="ROC curves", height=450,
        xaxis_title="False positive rate", yaxis_title="True positive rate",
    )
    st.plotly_chart(roc_fig, width="stretch")

    best = res.loc[res["f1"].idxmax()]
    st.success(
        f"**{best['model']}** wins on F1 ({best['f1']:.3f}) and ROC-AUC "
        f"({best['roc_auc']:.3f}). Logistic Regression reaches similar recall but "
        f"raises far more false alarms, which costs inspection hours."
    )

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Factors affecting machine failure")

    choice = st.selectbox("Model", imp["model"].unique(),
                          index=list(imp["model"].unique()).index(meta["model"]))
    sub = imp[imp["model"] == choice].sort_values("importance")

    fig = px.bar(
        sub, x="importance", y="label", orientation="h",
        title=f"Relative importance — {choice}", height=430,
    )
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
**Reading the chart.** Rotational speed, torque and the mechanical power derived
from them dominate — which matches the physics: power failure is defined on
torque times angular velocity. Tool wear is next, and the engineered
wear × torque term picks up overstrain risk. Air and process temperature matter
less individually than their difference, which is what drives heat dissipation
failure.

**Business reading.** Three of the top factors are set by the operator
(speed, torque, depth of cut), not by the machine. That means a meaningful share
of failures is addressable through parameter discipline and tool-change policy
rather than capital spend.
"""
    )

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Score a machine reading")
    st.caption(f"Using the best model: {meta['model']}")

    c1, c2 = st.columns(2)
    with c1:
        air = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process = st.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
        speed = st.slider("Rotational speed (rpm)", 1160, 2900, 1500, 10)
    with c2:
        torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
        wear = st.slider("Tool wear (min)", 0, 255, 100, 1)
        variant = st.selectbox("Product quality variant", ["H", "M", "L"], index=1)

    import numpy as np

    row = pd.DataFrame([{
        "air_temp_k": air,
        "process_temp_k": process,
        "rotational_speed_rpm": speed,
        "torque_nm": torque,
        "tool_wear_min": wear,
        "type_code": config.TYPE_MAP[variant],
        "temp_diff_k": process - air,
        "power_w": torque * speed * 2 * np.pi / 60,
        "wear_torque": wear * torque,
    }])[config.FEATURES]

    prob = float(model.predict_proba(row)[0, 1])

    m1, m2, m3 = st.columns(3)
    m1.metric("Failure probability", f"{prob:.1%}")
    m2.metric("Temperature difference", f"{process - air:.1f} K")
    m3.metric("Mechanical power", f"{row['power_w'].iloc[0]:,.0f} W")

    if prob >= 0.6:
        st.error("**Act this shift.** Schedule an intervention before the next batch.")
    elif prob >= 0.3:
        st.warning("**Inspect at the next planned break.**")
    else:
        st.success("**No action.** Continue running and monitor normally.")

    reasons = []
    if wear > 200:
        reasons.append("tool wear is past the 200-minute replacement threshold")
    if (process - air) < 8.6 and speed < 1380:
        reasons.append("low temperature difference at low speed — heat dissipation risk")
    p = row["power_w"].iloc[0]
    if p < 3500 or p > 9000:
        reasons.append(f"mechanical power ({p:,.0f} W) is outside the 3,500–9,000 W band")
    if wear * torque > {"L": 11000, "M": 12000, "H": 13000}[variant]:
        reasons.append(f"wear × torque is above the {variant}-variant overstrain limit")

    if reasons:
        st.markdown("**Contributing conditions:**")
        for r in reasons:
            st.markdown(f"- {r}")
