import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from fpdf import FPDF

st.set_page_config("🎬 Movie Intelligence Lab", layout="wide")
st.title("🎬 Movie Intelligence Lab")

file = st.file_uploader("📂 Upload Movie CSV", type="csv")

if not file:
    st.info("Please upload a CSV file")
    st.stop()

df = pd.read_csv(file)

# ===============================
# GLOBAL SUCCESS SCORING (FIX)
# ===============================
required_cols = {"rating", "votes", "revenue"}

df_score = None
if required_cols.issubset(df.columns):
    df_score = df.copy()

    for col in ["rating", "votes", "revenue"]:
        df_score[col] = (
            df_score[col] - df_score[col].min()
        ) / (df_score[col].max() - df_score[col].min())

    df_score["success_score"] = (
        0.5 * df_score["rating"]
        + 0.3 * df_score["revenue"]
        + 0.2 * df_score["votes"]
    )

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploration",
    "🎯 Success Scoring",
    "🤖 Predictive Model",
    "📄 Report"
])

# -------------------------------
# 📊 Exploration
# -------------------------------
with tab1:
    st.subheader("Exploratory Analysis")

    st.dataframe(df.head())

    if "year" in df.columns:
        fig = px.line(
            df.groupby("year").size().reset_index(name="count"),
            x="year", y="count",
            title="Movies Released per Year"
        )
        st.plotly_chart(fig, use_container_width=True)

    if {"genre", "rating"}.issubset(df.columns):
        fig = px.box(df, x="genre", y="rating", title="Ratings by Genre")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 🎯 Success Scoring
# -------------------------------
with tab2:
    st.subheader("Movie Success Scoring")

    if df_score is None:
        st.warning("CSV must contain rating, votes, and revenue columns.")
    else:
        st.metric(
            "Average Success Score",
            round(df_score["success_score"].mean(), 2)
        )

        fig = px.histogram(
            df_score,
            x="success_score",
            title="Success Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 🤖 Predictive Model
# -------------------------------
with tab3:
    st.subheader("Predictive Model (Explainable ML)")

    if df_score is None:
        st.warning("Cannot train model without success score.")
    else:
        X = df_score[["rating", "votes", "revenue"]]
        y = df_score["success_score"]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        model.fit(X, y)

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig = px.bar(
            importance,
            x="Feature", y="Importance",
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 📄 Report
# -------------------------------
with tab4:
    st.subheader("Auto-Generated Report")

    if df_score is None:
        st.warning("Report unavailable — missing required columns.")
    else:
        insights = [
            f"Average success score: {df_score['success_score'].mean():.2f}",
            f"Highest success score: {df_score['success_score'].max():.2f}",
            "Ratings have the highest impact on movie success"
        ]

        for i in insights:
            st.write("•", i)

        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(0, 10, "Movie Intelligence Lab Report", ln=True)
            pdf.ln(5)

            for i in insights:
                pdf.multi_cell(0, 8, i)

            pdf.output("report.pdf")

            st.download_button(
                "Download Report",
                data=open("report.pdf", "rb"),
                file_name="movie_report.pdf"
            )
