import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
def safe_text(text):
    if isinstance(text, str):
        return text.encode("latin-1", "replace").decode("latin-1")
    return str(text)
st.set_page_config(page_title="🎬 Movie Intelligence Lab", layout="wide")
st.title("🎬 Movie Intelligence Lab")

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
file = st.file_uploader("📂 Upload Movie CSV", type="csv")
if file is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

data = pd.read_csv(file)

# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
required_cols = {"rating", "votes", "revenue", "genre"}
if not required_cols.issubset(data.columns):
    st.error("CSV must contain rating, votes, revenue, and genre columns.")
    st.stop()

# -------------------------------------------------
# CLEANING
# -------------------------------------------------
data = data.dropna(subset=list(required_cols))

data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
data["votes"] = pd.to_numeric(data["votes"], errors="coerce")
data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
data = data.dropna(subset=["rating", "votes", "revenue"])

data["genre"] = data["genre"].astype(str).str.split("|")
data = data.explode("genre")
data["genre"] = data["genre"].str.strip()

# -------------------------------------------------
# NORMALIZATION (NO SHARED STATE)
# -------------------------------------------------
def normalize(col):
    return (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0

data["rating_n"] = normalize(data["rating"])
data["votes_n"] = normalize(data["votes"])
data["revenue_n"] = normalize(data["revenue"])

# -------------------------------------------------
# SUCCESS SCORE
# -------------------------------------------------
data["success_score"] = (
    0.5 * data["rating_n"]
    + 0.3 * data["revenue_n"]
    + 0.2 * data["votes_n"]
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
# TAB 1 — DASHBOARD
# -------------------------------------------------
with tab1:
    st.subheader("📊 Analyst Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", len(data))
    c2.metric("Avg Rating", round(data["rating"].mean(), 2))
    c3.metric("Median Revenue", f"${data['revenue'].median():,.0f}")
    c4.metric("Avg Success", round(data["success_score"].mean(), 3))

    genre_perf = (
        data.groupby("genre")
        .agg(
            avg_success=("success_score", "mean"),
            avg_rating=("rating", "mean"),
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
        hover_name="genre",
        title="Genre Performance Map"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 2 — SUCCESS SCORING
# -------------------------------------------------
with tab2:
    fig = px.histogram(data, x="success_score", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 3 — PREDICTIVE MODEL (NO X VARIABLE)
# -------------------------------------------------
with tab3:
    st.subheader("🤖 Success Score Prediction")

    features = data[["rating_n", "votes_n", "revenue_n"]]
    target = data["success_score"]

    f_train, f_test, t_train, t_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    reg.fit(f_train, t_train)

    preds = reg.predict(f_test)

    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2_score(t_test, preds):.3f}")
    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(t_test, preds)):.3f}")

    importance_df = pd.DataFrame({
        "Feature": ["rating", "votes", "revenue"],
        "Importance": reg.feature_importances_
    })

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 4 — REVENUE CLASSIFICATION
# -------------------------------------------------
with tab4:
    q40 = data["revenue"].quantile(0.40)
    q75 = data["revenue"].quantile(0.75)

    data["revenue_class"] = np.select(
        [
            data["revenue"] >= q75,
            data["revenue"] >= q40
        ],
        ["Hit", "Average"],
        default="Flop"
    )

    cls_features = data[["rating_n", "votes_n"]]
    cls_target = data["revenue_class"]

    le = LabelEncoder()
    cls_target_enc = le.fit_transform(cls_target)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    clf.fit(cls_features, cls_target_enc)

    pred_labels = le.inverse_transform(clf.predict(cls_features))

    fig = px.pie(
        names=pd.Series(pred_labels).value_counts().index,
        values=pd.Series(pred_labels).value_counts().values,
        title="Revenue Outcome Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 5 — GENRE-WISE PREDICTION
# -------------------------------------------------
with tab5:
    st.subheader("🎭 Genre-wise Success Prediction")

    valid_genres = [
        g for g in data["genre"].unique()
        if len(data[data["genre"] == g]) >= 20
    ]

    if not valid_genres:
        st.warning("Not enough data per genre.")
        st.stop()

    selected_genre = st.selectbox("Select Genre", valid_genres)

    gdf = data[data["genre"] == selected_genre]

    g_features = gdf[["rating_n", "votes_n", "revenue_n"]]
    g_target = gdf["success_score"]

    g_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )
    g_model.fit(g_features, g_target)

    r = st.slider("Rating", 0.0, 10.0, 7.0)
    v = st.number_input("Votes", 1000, 1_000_000, 50000)
    rev = st.number_input("Revenue ($)", 1e6, 1e9, 5e7)

    inp = pd.DataFrame([{
        "rating_n": (r - data["rating"].min()) / (data["rating"].max() - data["rating"].min()),
        "votes_n": (v - data["votes"].min()) / (data["votes"].max() - data["votes"].min()),
        "revenue_n": (rev - data["revenue"].min()) / (data["revenue"].max() - data["revenue"].min())
    }])
    # ---------- BUILD PDF ----------
pdf = PDF()
pdf.add_page()

pdf.set_font("Arial", "B", 14)
pdf.cell(
    0, 10,
    safe_text("Movie Intelligence Lab - Analysis Report"),
    ln=True
)
pdf.ln(5)

pdf.set_font("Arial", size=11)
for i in insights:
    pdf.multi_cell(0, 8, safe_text(f"- {i}"))
pdf.ln(5)

pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, safe_text("Success Score Distribution"), ln=True)
pdf.image("success_dist.png", w=170)
pdf.ln(5)

if len(model_df) >= 10:
    pdf.cell(0, 10, safe_text("Feature Importance"), ln=True)
    pdf.image("feature_importance.png", w=170)

pdf.output("movie_analysis_report.pdf")


if st.button("Predict Success"):
    score = g_model.predict(inp)[0]
    st.success(f"🎬 Predicted Success Score: {score:.3f}")
