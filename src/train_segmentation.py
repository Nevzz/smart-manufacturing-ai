"""Module 2 -- Machine Segmentation.

Groups the 10,000 operating records into clusters with K-Means, projects them
to 2D with PCA for visualisation, and profiles each segment (including its
failure rate, which is what makes the clusters actionable).

Run:  python src/train_segmentation.py
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from src import config, database
except ImportError:
    import config
    import database


def choose_k(X_scaled, k_range=range(2, 9)) -> pd.DataFrame:
    """Elbow + silhouette scan, saved so the app can show it."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X_scaled, labels, sample_size=3000,
                                               random_state=config.RANDOM_STATE),
            }
        )
    return pd.DataFrame(rows)


NAME_DIMS = ["rotational_speed_rpm", "torque_nm", "tool_wear_min"]

DIM_LABELS = {
    ("rotational_speed_rpm", "high"): "High-speed operation",
    ("rotational_speed_rpm", "low"): "Low-speed operation",
    ("torque_nm", "high"): "Heavy-load operation",
    ("torque_nm", "low"): "Light-load operation",
    ("tool_wear_min", "high"): "Worn tooling",
    ("tool_wear_min", "low"): "Fresh tooling",
}


def name_segments(profile: pd.DataFrame) -> list[str]:
    """Label each cluster by the dimension on which its centroid deviates most
    from the average centroid. If two clusters share a primary label, the
    second-strongest dimension is appended to keep the names distinct."""
    centroids = profile[NAME_DIMS]
    spread = centroids.std(ddof=0).replace(0, 1)
    z = (centroids - centroids.mean()) / spread

    pairs = []
    for _, row in z.iterrows():
        ranked = row.abs().sort_values(ascending=False).index.tolist()
        primary = DIM_LABELS[(ranked[0], "high" if row[ranked[0]] > 0 else "low")]
        secondary = DIM_LABELS[(ranked[1], "high" if row[ranked[1]] > 0 else "low")]
        pairs.append((primary, secondary))

    counts: dict[str, int] = {}
    for primary, _ in pairs:
        counts[primary] = counts.get(primary, 0) + 1

    return [
        f"{p} / {s.lower()}" if counts[p] > 1 else p
        for p, s in pairs
    ]


def main() -> None:
    print("=" * 60)
    print("MODULE 2 -- Machine Segmentation")
    print("=" * 60)

    df = database.load_sensor_data()
    X = df[config.CLUSTER_FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scan = choose_k(X_scaled)
    scan.to_csv(config.DATA_PROCESSED / "kmeans_scan.csv", index=False)
    print("\nk-selection scan:")
    for _, r in scan.iterrows():
        print(f"  k={int(r['k'])}  inertia {r['inertia']:>9,.0f}  "
              f"silhouette {r['silhouette']:.3f}")

    k = config.N_CLUSTERS
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE)
    df["cluster"] = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    df["pca_1"], df["pca_2"] = coords[:, 0], coords[:, 1]
    var = pca.explained_variance_ratio_
    print(f"\nPCA: PC1 {var[0]:.1%} + PC2 {var[1]:.1%} = "
          f"{var.sum():.1%} of variance retained")

    # ---- profile each segment ------------------------------------------
    profile = (
        df.groupby("cluster")[config.CLUSTER_FEATURES].mean().round(1)
    )
    profile["machines"] = df.groupby("cluster").size()
    profile["failure_rate"] = df.groupby("cluster")[config.TARGET].mean().round(4)
    profile["segment_name"] = name_segments(profile)
    profile = profile.reset_index()

    df["segment_name"] = df["cluster"].map(
        dict(zip(profile["cluster"], profile["segment_name"]))
    )

    profile.to_csv(config.DATA_PROCESSED / "segment_profiles.csv", index=False)
    df.to_csv(config.SEGMENTS_CSV, index=False)
    joblib.dump({"kmeans": km, "scaler": scaler, "pca": pca},
                config.MODELS / "segmentation.pkl")
    database.save_table(df, "machine_segments")

    print(f"\nSegment profiles (k={k}):")
    for _, r in profile.iterrows():
        print(f"  Cluster {int(r['cluster'])}: {r['segment_name']:<26} "
              f"{int(r['machines']):>5,} records | failure rate {r['failure_rate']:.2%}")
        print(f"      speed {r['rotational_speed_rpm']:.0f} rpm, "
              f"torque {r['torque_nm']:.1f} Nm, wear {r['tool_wear_min']:.0f} min")

    worst = profile.loc[profile["failure_rate"].idxmax()]
    print(f"\nHighest-risk segment: {worst['segment_name']} "
          f"({worst['failure_rate']:.2%} failure rate)")
    print("\nDone. Next: python src/train_vision.py\n")


if __name__ == "__main__":
    main()
