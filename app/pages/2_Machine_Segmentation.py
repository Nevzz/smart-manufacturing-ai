import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config

st.set_page_config(page_title="Machine Segmentation", page_icon="🧩", layout="wide")
st.title("🧩 Machine Segmentation")
st.caption("K-Means clustering with PCA visualisation")

PROFILES = config.DATA_PROCESSED / "segment_profiles.csv"
SCAN = config.DATA_PROCESSED / "kmeans_scan.csv"

if not config.SEGMENTS_CSV.exists():
    st.warning("Segmentation not run yet. Run `python src/train_segmentation.py` first.")
    st.stop()


@st.cache_data
def load():
    return (
        pd.read_csv(config.SEGMENTS_CSV),
        pd.read_csv(PROFILES),
        pd.read_csv(SCAN),
    )


df, profiles, scan = load()

tab1, tab2, tab3 = st.tabs(["Segment map", "Segment profiles", "Choosing k"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.subheader("Operating regimes projected onto two principal components")

    colour_by = st.radio(
        "Colour points by", ["Segment", "Machine failure"], horizontal=True
    )

    sample = df.sample(min(4000, len(df)), random_state=config.RANDOM_STATE)

    if colour_by == "Segment":
        fig = px.scatter(
            sample, x="pca_1", y="pca_2", color="segment_name",
            hover_data=["rotational_speed_rpm", "torque_nm", "tool_wear_min"],
            opacity=0.6, height=560, title="PCA projection coloured by segment",
        )
    else:
        sample = sample.copy()
        sample["outcome"] = sample[config.TARGET].map({0: "No failure", 1: "Failure"})
        fig = px.scatter(
            sample, x="pca_1", y="pca_2", color="outcome",
            color_discrete_map={"No failure": "#c9d1d9", "Failure": "#d62728"},
            opacity=0.7, height=560, title="PCA projection coloured by outcome",
        )

    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2", legend_title="")
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "The two components retain about 75% of the variance in the five sensor "
        "readings, so the picture is a fair but not complete summary."
    )

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("What each segment looks like")

    show = profiles.copy()
    show["failure_rate"] = (show["failure_rate"] * 100).round(2)
    show = show.rename(columns={
        "cluster": "Cluster",
        "segment_name": "Segment",
        "machines": "Records",
        "failure_rate": "Failure rate (%)",
        "air_temp_k": "Air temp (K)",
        "process_temp_k": "Process temp (K)",
        "rotational_speed_rpm": "Speed (rpm)",
        "torque_nm": "Torque (Nm)",
        "tool_wear_min": "Tool wear (min)",
    })
    order = ["Cluster", "Segment", "Records", "Failure rate (%)", "Speed (rpm)",
             "Torque (Nm)", "Tool wear (min)", "Air temp (K)", "Process temp (K)"]
    st.dataframe(show[order], width="stretch", hide_index=True)

    overall = df[config.TARGET].mean() * 100
    bar = px.bar(
        show.sort_values("Failure rate (%)"),
        x="Failure rate (%)", y="Segment", orientation="h",
        title="Failure rate by segment", height=380,
        color="Failure rate (%)", color_continuous_scale="Reds",
    )
    bar.add_vline(x=overall, line_dash="dot", line_color="grey",
                  annotation_text=f"plant average {overall:.2f}%")
    bar.update_layout(yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(bar, width="stretch")

    worst = show.loc[show["Failure rate (%)"].idxmax()]
    st.info(
        f"**{worst['Segment']}** carries the highest risk at "
        f"{worst['Failure rate (%)']:.2f}%, against a plant average of "
        f"{overall:.2f}%. Concentrating inspection effort on machines currently "
        "in this regime targets maintenance by risk rather than by calendar."
    )

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("How the number of clusters was chosen")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scan["k"], y=scan["inertia"], name="Inertia (elbow)",
                             mode="lines+markers"))
    fig.add_trace(go.Scatter(x=scan["k"], y=scan["silhouette"], name="Silhouette",
                             mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        height=430,
        xaxis_title="k (number of clusters)",
        yaxis=dict(title="Inertia"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
    )
    fig.add_vline(x=config.N_CLUSTERS, line_dash="dot", line_color="green",
                  annotation_text=f"chosen k = {config.N_CLUSTERS}")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"""
Inertia falls smoothly with no sharp elbow, and silhouette scores sit around
0.22–0.26 across the range — the sensor readings form one continuous cloud
rather than well-separated groups. **k = {config.N_CLUSTERS}** was chosen because it
produces segments that are interpretable on the shop floor and that differ
meaningfully in failure rate, which is the property the business actually needs.

This is worth stating honestly in the report: the clusters are a useful way of
organising operating conditions, not evidence of naturally distinct machine
populations.
"""
    )
