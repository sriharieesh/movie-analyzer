import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
try:
    import shap
    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    SHAP_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🎬 Movie Intelligence Lab", layout="wide")
st.title("🎬 Movie Intelligence Lab")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_text(text):
    return text.encode("latin-1", "replace").decode("latin-1")

norm_params = {}

def fit_norm(col):
    mn, mx = col.min(), col.max()
    norm_params[col.name] = (mn, mx)
    return (col - mn) / (mx - mn)

def apply_norm(val, name):
    mn, mx = norm_params[name]
    return (val - mn) / (mx - mn)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
file = st.file_uploader("📂 Upload Movie CSV", type="csv")
if not file:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(file)

# -------------------------------------------------
# BASIC CLEANING
# -------------------------------------------------
required_cols = {"rating", "votes", "revenue", "genre"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain rating, votes, revenue, genre columns.")
    st.stop()

df = df.dropna(subset=required_cols)

df["genre"] = df["genre"].astype(str).str.split("|")
df = df.explode("genre")
df["genre"] = df["genre"].str.strip()

# -------------------------------------------------
# NORMALIZATION + SUCCESS SCORE
# -------------------------------------------------
df["rating"] = pd.to_numeric(df["rating"])
df["votes"] = pd.to_numeric(df["votes"])
df["revenue"] = pd.to_numeric(df["revenue"])

df["rating_n"] = fit_norm(df["rating"])
df["votes_n"] = fit_norm(df["votes"])
df["revenue_n"] = fit_norm(df["revenue"])

df["success_score"] = (
    0.5 * df["rating_n"] +
    0.3 * df["revenue_n"] +
    0.2 * df["votes_n"]
)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🎯 Success Scoring",
    "🤖 Predictive Model",
    "💰 Revenue Classification",
    "🎭 Genre-wise Prediction"
])

# -------------------------------------------------
# TAB 1 — ANALYST DASHBOARD
# -------------------------------------------------
with tab1:
    st.subheader("📊 Analyst Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", len(df))
    c2.metric("Avg Rating", round(df["rating"].mean(), 2))
    c3.metric("Median Revenue", f"${df['revenue'].median():,.0f}")
    c4.metric("Avg Success", round(df["success_score"].mean(), 3))

    genre_perf = (
        df.groupby("genre")
        .agg(
            avg_success=("success_score", "mean"),
            avg_rating=("rating", "mean"),
            avg_revenue=("revenue", "mean"),
            count=("genre", "count")
        )
        .query("count >= 10")
        .reset_index()
    )

    fig = px.scatter(
        genre_perf,
        x="avg_rating",
        y="avg_success",
        size="count",
        color="avg_revenue",
        hover_name="genre",
        title="Genre Performance Landscape"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 2 — SUCCESS SCORING
# -------------------------------------------------
with tab2:
    st.subheader("🎯 Success Score Distribution")

    fig = px.histogram(df, x="success_score", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 3 — PREDICTIVE MODEL + SHAP
# -------------------------------------------------
st.subheader("🔍 Explainable AI")

if SHAP_AVAILABLE:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)
    plt.close()
else:
    st.info(
        "Explainable AI (SHAP) is disabled because the SHAP library "
        "is not installed in this environment.\n\n"
        "To enable it locally, run:\n"
        "`pip install shap`"
    )



# -------------------------------------------------
# TAB 4 — REVENUE CLASSIFICATION
# -------------------------------------------------
with tab4:
    st.subheader("💰 Revenue Classification")

    q40 = df["revenue"].quantile(0.40)
    q75 = df["revenue"].quantile(0.75)

    df["revenue_class"] = np.select(
        [
            df["revenue"] >= q75,
            df["revenue"] >= q40
        ],
        ["Hit", "Average"],
        default="Flop"
    )

    X_cls = df[["rating_n", "votes_n"]]
    y_cls = df["revenue_class"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_cls)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    clf.fit(X_cls, y_enc)

    preds = le.inverse_transform(clf.predict(X_cls))

    fig = px.pie(
        names=pd.Series(preds).value_counts().index,
        values=pd.Series(preds).value_counts().values,
        title="Revenue Outcome Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 5 — GENRE-WISE SUCCESS PREDICTION
# -------------------------------------------------
with tab5:
    st.subheader("🎭 Genre-wise Success Prediction")

    genre_models = {}

    for g in df["genre"].unique():
        gdf = df[df["genre"] == g]
        if len(gdf) < 20:
            continue

        Xg = gdf[["rating_n", "votes_n", "revenue_n"]]
        yg = gdf["success_score"]

        m = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            random_state=42
        )
        m.fit(Xg, yg)
        genre_models[g] = m

    genre = st.selectbox("Select Genre", list(genre_models.keys()))

    rating = st.slider("Rating", 0.0, 10.0, 7.0)
    votes = st.number_input("Votes", 1000, 1_000_000, 50000)
    revenue = st.number_input("Revenue ($)", 1e6, 1e9, 5e7)

    if st.button("Predict Success"):
        inp = pd.DataFrame([{
            "rating_n": apply_norm(rating, "rating"),
            "votes_n": apply_norm(votes, "votes"),
            "revenue_n": apply_norm(revenue, "revenue")
        }])

        pred = genre_models[genre].predict(inp)[0]
        st.success(f"🎬 Predicted Success Score: {pred:.3f}")
