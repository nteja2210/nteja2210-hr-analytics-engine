"""Intelligent HR Analytics Engine - Streamlit dashboard.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explain import build_explainer, explain_employee, global_importance
from src.preprocess import TARGET, load_data
from src.segment import SEG_FEATURES, elbow_analysis, segment_employees

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Intelligent HR Analytics Engine", page_icon="👥", layout="wide")


# ---------------------------------------------------------------- cached loaders
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        from src.train import main as train_main
        train_main()
    bundle = joblib.load(MODEL_PATH)
    X = get_data().drop(columns=[TARGET])
    explainer = build_explainer(bundle["pipeline"], X.sample(min(150, len(X)), random_state=1))
    return bundle, explainer


@st.cache_data
def get_scored_data() -> pd.DataFrame:
    bundle, _ = get_model()
    df = get_data().copy()
    df["Risk"] = bundle["pipeline"].predict_proba(df.drop(columns=[TARGET]))[:, 1]
    return df


def risk_band(p: float) -> str:
    return "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"


df = get_data()
bundle, explainer = get_model()
pipe = bundle["pipeline"]

# ---------------------------------------------------------------- sidebar
st.sidebar.title("👥 HR Analytics Engine")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Attrition Analysis", "Predict & Explain", "Employee Segments", "Model Performance"],
)
st.sidebar.caption(f"Model in use: **{bundle['name']}**")

# ================================================================ 1. OVERVIEW
if page == "Overview":
    st.title("Workforce Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headcount", f"{len(df):,}")
    c2.metric("Attrition rate", f"{df[TARGET].mean() * 100:.1f}%")
    c3.metric("Avg monthly income", f"${df['MonthlyIncome'].mean():,.0f}")
    c4.metric("Avg tenure (yrs)", f"{df['YearsAtCompany'].mean():.1f}")

    left, right = st.columns(2)
    dept = df.groupby("Department").size().reset_index(name="Employees")
    left.plotly_chart(px.pie(dept, names="Department", values="Employees", hole=0.45,
                             title="Headcount by department"), width="stretch")
    right.plotly_chart(px.histogram(df, x="Age", nbins=25, title="Age distribution"),
                       width="stretch")

    st.subheader("Highest-risk employees right now")
    scored = get_scored_data()
    top = scored[scored[TARGET] == 0].sort_values("Risk", ascending=False).head(10)
    show = top[["Department", "JobRole", "Age", "MonthlyIncome", "OverTime", "JobSatisfaction", "Risk"]].copy()
    show["Risk"] = (show["Risk"] * 100).round(1).astype(str) + "%"
    st.dataframe(show, width="stretch")
    st.caption("Current employees (Attrition = No) ranked by predicted probability of leaving.")

# ================================================================ 2. ATTRITION ANALYSIS
elif page == "Attrition Analysis":
    st.title("Attrition Analysis")
    depts = st.sidebar.multiselect("Department filter", sorted(df["Department"].unique()),
                                   default=sorted(df["Department"].unique()))
    view = df[df["Department"].isin(depts)]
    if view.empty:
        st.warning("Select at least one department.")
        st.stop()

    st.metric("Attrition rate (filtered)", f"{view[TARGET].mean() * 100:.1f}%")

    def rate_chart(col, title, horizontal=False):
        g = view.groupby(col)[TARGET].mean().mul(100).round(1).reset_index()
        g.columns = [col, "Attrition %"]
        g = g.sort_values("Attrition %", ascending=horizontal)
        if horizontal:
            return px.bar(g, x="Attrition %", y=col, orientation="h", title=title)
        return px.bar(g, x=col, y="Attrition %", title=title)

    a, b = st.columns(2)
    a.plotly_chart(rate_chart("OverTime", "Attrition % by overtime"), width="stretch")
    b.plotly_chart(rate_chart("Department", "Attrition % by department"), width="stretch")
    st.plotly_chart(rate_chart("JobRole", "Attrition % by job role", horizontal=True),
                    width="stretch")

    view = view.assign(AgeGroup=pd.cut(view["Age"], [17, 25, 35, 45, 55, 100],
                                       labels=["18-25", "26-35", "36-45", "46-55", "55+"]))
    c, d = st.columns(2)
    c.plotly_chart(px.bar(
        view.groupby("AgeGroup", observed=True)[TARGET].mean().mul(100).round(1).reset_index(),
        x="AgeGroup", y=TARGET, title="Attrition % by age group", labels={TARGET: "Attrition %"}),
        width="stretch")
    d.plotly_chart(px.box(view.assign(Status=view[TARGET].map({0: "Stayed", 1: "Left"})),
                          x="Status", y="MonthlyIncome", title="Monthly income: stayed vs left"),
                   width="stretch")

    st.subheader("What drives attrition? (global SHAP importance)")
    imp = global_importance(pipe, explainer, df.drop(columns=[TARGET]).sample(min(400, len(df)), random_state=1))
    st.plotly_chart(px.bar(imp.iloc[::-1], x="importance", y="feature", orientation="h",
                           title="Mean |SHAP value| - top 15 features"), width="stretch")

# ================================================================ 3. PREDICT & EXPLAIN
elif page == "Predict & Explain":
    st.title("Predict Attrition Risk for an Employee")
    st.caption("Enter employee details. Fields not shown use the company average / most common value.")

    X = df.drop(columns=[TARGET])
    defaults = {
        c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode()[0])
        for c in X.columns
    }

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 18, 60, 35)
        income = c1.number_input("Monthly income ($)", 1000, 20000, int(defaults["MonthlyIncome"]), step=100)
        dept = c1.selectbox("Department", sorted(X["Department"].unique()))
        role = c1.selectbox("Job role", sorted(X["JobRole"].unique()))
        overtime = c2.selectbox("Works overtime?", ["No", "Yes"])
        travel = c2.selectbox("Business travel", sorted(X["BusinessTravel"].unique()), index=2)
        marital = c2.selectbox("Marital status", sorted(X["MaritalStatus"].unique()))
        distance = c2.slider("Distance from home (km)", 1, 30, 8)
        job_sat = c3.slider("Job satisfaction (1-4)", 1, 4, 3)
        env_sat = c3.slider("Environment satisfaction (1-4)", 1, 4, 3)
        wlb = c3.slider("Work-life balance (1-4)", 1, 4, 3)
        years = c3.slider("Years at company", 0, 40, 5)
        since_promo = c3.slider("Years since last promotion", 0, 15, 2)
        stock = c1.selectbox("Stock option level", [0, 1, 2, 3])
        companies = c2.slider("Companies worked before", 0, 9, 2)
        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        row = pd.DataFrame([defaults])
        updates = {
            "Age": age, "MonthlyIncome": income, "Department": dept, "JobRole": role,
            "OverTime": overtime, "BusinessTravel": travel, "MaritalStatus": marital,
            "DistanceFromHome": distance, "JobSatisfaction": job_sat,
            "EnvironmentSatisfaction": env_sat, "WorkLifeBalance": wlb,
            "YearsAtCompany": years, "YearsSinceLastPromotion": since_promo,
            "StockOptionLevel": stock, "NumCompaniesWorked": companies,
        }
        for k, v in updates.items():
            row[k] = v
        # keep tenure columns logically consistent
        row["YearsInCurrentRole"] = min(int(defaults["YearsInCurrentRole"]), years)
        row["YearsWithCurrManager"] = min(int(defaults["YearsWithCurrManager"]), years)
        row["TotalWorkingYears"] = max(int(defaults["TotalWorkingYears"]), years)
        row = row[X.columns]

        prob = float(pipe.predict_proba(row)[0, 1])
        band = risk_band(prob)

        g1, g2 = st.columns([1, 1.4])
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": f"Attrition risk: {band}"},
            gauge={"axis": {"range": [0, 100]},
                   "steps": [{"range": [0, 35], "color": "#d1fae5"},
                             {"range": [35, 60], "color": "#fef3c7"},
                             {"range": [60, 100], "color": "#fee2e2"}],
                   "bar": {"color": "#1f2937"}}))
        fig.update_layout(height=300, margin=dict(t=60, b=10))
        g1.plotly_chart(fig, width="stretch")

        exp = explain_employee(pipe, explainer, row, top_n=8)
        exp["Effect"] = exp["shap_value"].apply(lambda v: "Raises risk" if v > 0 else "Lowers risk")
        g2.plotly_chart(px.bar(exp.iloc[::-1], x="shap_value", y="feature", orientation="h",
                               color="Effect",
                               color_discrete_map={"Raises risk": "#dc2626", "Lowers risk": "#2563eb"},
                               title="Why this prediction? (SHAP)"), width="stretch")

        st.subheader("Suggested HR actions")
        tips = []
        raising = set(exp.loc[exp["shap_value"] > 0, "feature"])
        if "OverTime" in raising or any(f.startswith("OverTime") for f in raising):
            tips.append("Reduce overtime load or offer compensatory time off.")
        if "JobSatisfaction" in raising:
            tips.append("Hold a 1-on-1 to understand job-satisfaction concerns; consider role enrichment.")
        if "WorkLifeBalance" in raising:
            tips.append("Offer flexible hours / hybrid work.")
        if "YearsSinceLastPromotion" in raising:
            tips.append("Review the career path and promotion timeline.")
        if "MonthlyIncome" in raising or "StockOptionLevel" in raising:
            tips.append("Benchmark pay and consider a retention bonus or stock options.")
        if "DistanceFromHome" in raising:
            tips.append("Consider remote days or commute support.")
        if "EnvironmentSatisfaction" in raising:
            tips.append("Survey the team about workplace environment issues.")
        if not tips:
            tips.append("No major risk drivers detected - continue regular check-ins.")
        for t in tips:
            st.write("• " + t)

# ================================================================ 4. SEGMENTS
elif page == "Employee Segments":
    st.title("Employee Segmentation (KMeans)")
    k = st.sidebar.slider("Number of clusters (k)", 2, 6, 4)
    seg_df, profile = segment_employees(df, k=k)

    st.subheader("Segment profiles")
    st.dataframe(profile, width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.scatter(seg_df, x="Age", y="MonthlyIncome", color="Segment",
                              title="Segments by age and income", opacity=0.7),
                   width="stretch")
    b.plotly_chart(px.bar(profile, x="Segment", y="Attrition_Rate", color="Segment",
                          title="Attrition % by segment"), width="stretch")

    st.subheader("Choosing k: elbow & silhouette")
    el = elbow_analysis(df)
    e1, e2 = st.columns(2)
    e1.plotly_chart(px.line(el, x="k", y="inertia", markers=True, title="Elbow curve (inertia)"),
                    width="stretch")
    e2.plotly_chart(px.line(el, x="k", y="silhouette", markers=True, title="Silhouette score"),
                    width="stretch")

# ================================================================ 5. MODEL PERFORMANCE
else:
    st.title("Model Performance")
    m = json.loads(METRICS_PATH.read_text())
    res = pd.DataFrame(m["results"]).T.reset_index().rename(columns={"index": "Model"})
    st.success(f"Best model (highest ROC-AUC): **{m['best']}**")
    st.dataframe(res, width="stretch")
    st.caption("Recall and F1 matter more than accuracy here because only ~16-18% of employees leave.")
    fig_dir = ROOT / "reports" / "figures"
    c1, c2 = st.columns(2)
    if (fig_dir / "roc_curves.png").exists():
        c1.image(str(fig_dir / "roc_curves.png"), caption="ROC curves")
    if (fig_dir / "confusion_matrix.png").exists():
        c2.image(str(fig_dir / "confusion_matrix.png"), caption="Confusion matrix (best model)")
