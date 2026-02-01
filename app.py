import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from fpdf import FPDF

# ---------------- CONFIG ----------------
st.set_page_config("🎬 Movie Intelligence Lab", layout="wide")
st.title("🎬 Movie Intelligence Lab")

# ---------------- UPLOAD ----------------
file = st.file_uploader("📂 Upload Movie CSV", type="csv")

if not file:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(file)

# ---------------- SUCCESS SCORE (SAFE) ----------------
required_cols = {"rating", "votes", "revenue"}
df_score = None

if required_cols.issubset(df.columns):
    df_score = df.copy()

    for col in ["rating", "votes", "revenue"]:
        min_val = df_score[col].min()
        max_val = df_score[col].max()

        # 🚑 divide-by-zero protection
        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            df_score[col] = 0.0
        else:
            df_score[col] = (df_score[col] - min_val) / (max_val - min_val)

    df_score["success_score"] = (
        0.5 * df_score["rating"]
        + 0.3 * df_score["revenue"]
        + 0.2 * df_score["votes"]
    )

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploration",
    "🎯 Success Scoring",
    "🤖 Predictive Model",
    "📄 Report"
])

# ---------------- EXPLORATION ----------------
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
        fig = px.box(df, x="genre", y="rating", title="Rating Distribution by Genre")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- SUCCESS SCORING ----------------
with tab2:
    st.subheader("🎯 Movie Success Scoring")

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

# ---------------- PREDICTIVE MODEL ----------------
with tab3:
    st.subheader("🤖 Predictive Model (Explainable)")

    if df_score is None:
        st.warning("Success score not available.")
    else:
        model_df = df_score[
            ["rating", "votes", "revenue", "success_score"]
        ].dropna()

        if len(model_df) < 10:
            st.warning("Not enough clean data to train model.")
        else:
            X = model_df[["rating", "votes", "revenue"]]
            y = model_df["success_score"]

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
                title="Feature Importance on Movie Success"
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------- REPORT ----------------
with tab4:
    st.subheader("📄 Auto-Generated Report")

    if df_score is None:
        st.warning("Report unavailable.")
    else:
        insights = [
            f"Average success score: {df_score['success_score'].mean():.2f}",
            f"Highest success score: {df_score['success_score'].max():.2f}",
            "Ratings have the strongest influence on success"
        ]

        for i in insights:
            st.write("•", i)

        if st.button("📥 Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(0, 10, "Movie Intelligence Lab Report", ln=True)
            pdf.ln(5)

            for i in insights:
                pdf.multi_cell(0, 8, i)

            pdf.output("movie_report.pdf")

            st.download_button(
                "⬇️ Download PDF",
                data=open("movie_report.pdf", "rb"),
                file_name="movie_report.pdf"
            )
