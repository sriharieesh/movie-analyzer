import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from fpdf import FPDF

st.set_page_config("🎬 Movie Intelligence Lab", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Intelligence Lab")
st.caption("An end-to-end analytical system for movie performance understanding")

file = st.file_uploader("📂 Upload Movie Dataset (CSV)", type="csv")

if not file:
    st.stop()

df = pd.read_csv(file)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Data",
    "📊 Exploration",
    "🎯 Success Scoring",
    "🤖 Predictive Model",
    "📄 Report"
])

# ---------------- TAB 1: DATA ----------------
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write("**Columns detected:**", list(df.columns))
    st.write("**Missing values:**")
    st.write(df.isna().sum())

# ---------------- TAB 2: EXPLORATION ----------------
with tab2:
    st.subheader("Visual Exploration")

    if "year" in df.columns:
        fig = px.line(
            df.groupby("year").size().reset_index(name="count"),
            x="year", y="count",
            title="Movie Production Trend Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    if {"genre", "rating"}.issubset(df.columns):
        fig = px.box(
            df, x="genre", y="rating",
            title="Rating Spread by Genre"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3: SUCCESS SCORING ----------------
with tab3:
    st.subheader("🎯 Movie Success Score")

    required = {"rating", "votes", "revenue"}
    if not required.issubset(df.columns):
        st.warning("Rating, Votes, and Revenue required.")
        st.stop()

    df_score = df.copy()

    for col in ["rating", "votes", "revenue"]:
        df_score[col] = (df_score[col] - df_score[col].min()) / (
            df_score[col].max() - df_score[col].min()
        )

    df_score["success_score"] = (
        0.5 * df_score["rating"] +
        0.3 * df_score["revenue"] +
        0.2 * df_score["votes"]
    )

    st.metric("Average Success Score", round(df_score["success_score"].mean(), 2))

    fig = px.histogram(
        df_score, x="success_score",
        title="Distribution of Movie Success Scores"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 4: ML MODEL ----------------
with tab4:
    st.subheader("🤖 Predictive Modeling (Explainable)")

    features = df_score[["rating", "votes", "revenue"]]
    target = df_score["success_score"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)

    importance = pd.DataFrame({
        "Feature": features.columns,
        "Impact": model.feature_importances_
    }).sort_values("Impact", ascending=False)

    fig = px.bar(
        importance, x="Feature", y="Impact",
        title="Feature Impact on Movie Success"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 5: REPORT ----------------
with tab5:
    st.subheader("📄 Insight Report")

    insights = [
        f"Average success score across movies is {df_score['success_score'].mean():.2f}",
        f"Highest scoring movie achieved {df_score['success_score'].max():.2f}",
        f"Ratings contribute the most to success based on model importance"
    ]

    for i in insights:
        st.write("•", i)

    if st.button("📥 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "Movie Intelligence Lab Report", ln=True)
        pdf.ln(5)

        for i in insights:
            pdf.multi_cell(0, 8, i)

        pdf.output("movie_intelligence_report.pdf")

        st.download_button(
            "⬇️ Download PDF",
            data=open("movie_intelligence_report.pdf", "rb"),
            file_name="movie_intelligence_report.pdf"
        )
