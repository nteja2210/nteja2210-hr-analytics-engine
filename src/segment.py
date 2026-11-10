"""Employee segmentation with KMeans (+ elbow / silhouette analysis)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = [
    "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
    "WorkLifeBalance", "YearsSinceLastPromotion",
]


def elbow_analysis(df: pd.DataFrame, k_range=range(2, 9)) -> pd.DataFrame:
    X = StandardScaler().fit_transform(df[SEG_FEATURES])
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def _name_clusters(centers: pd.DataFrame) -> dict:
    """Give each cluster a business-friendly name from its centroid profile."""
    z = (centers - centers.mean()) / centers.std(ddof=0).replace(0, 1)
    rules = [
        ("Senior Experts", z["MonthlyIncome"] + z["YearsAtCompany"] + z["Age"]),
        ("Early-Career Talent", -(z["Age"] + z["YearsAtCompany"] + z["MonthlyIncome"])),
        ("Disengaged / Stalled", -(z["JobSatisfaction"] + z["WorkLifeBalance"]) + z["YearsSinceLastPromotion"]),
        ("Engaged Core Team", z["JobSatisfaction"] + z["WorkLifeBalance"]),
    ]
    names, taken = {}, set()
    for label, score in rules:
        for cid in score.sort_values(ascending=False).index:
            if cid not in taken:
                names[cid] = label
                taken.add(cid)
                break
    for cid in centers.index:                      # extra clusters if k > 4
        names.setdefault(cid, f"Segment {cid + 1}")
    return names


def segment_employees(df: pd.DataFrame, k: int = 4):
    """Return (df with 'Segment' column, per-segment profile table)."""
    X = StandardScaler().fit_transform(df[SEG_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    out = df.copy()
    out["_cluster"] = km.labels_
    centers = out.groupby("_cluster")[SEG_FEATURES].mean()
    names = _name_clusters(centers)
    out["Segment"] = out["_cluster"].map(names)

    profile = out.groupby("Segment").agg(
        Employees=("Age", "size"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("MonthlyIncome", "mean"),
        Avg_Tenure=("YearsAtCompany", "mean"),
        Avg_JobSat=("JobSatisfaction", "mean"),
        Avg_WLB=("WorkLifeBalance", "mean"),
        Attrition_Rate=("Attrition", "mean") if "Attrition" in out else ("Age", "size"),
    ).round(2)
    if "Attrition" in out:
        profile["Attrition_Rate"] = (profile["Attrition_Rate"] * 100).round(1)
    return out.drop(columns="_cluster"), profile.reset_index()
