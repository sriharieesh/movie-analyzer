import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from fpdf import FPDF

# -------------------------------------------------
# HELPER FUNCTION (UNICODE SAFE FOR PDF)
# -------------------------------------------------
def safe_text(text):
    if isinstance(text, str):
        return text.encode("latin-1", "replace").decode("latin-1")
    return str(text)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🎬 Movie Intelligence Lab", layout="wide")
st.title("🎬 Movie Intelligence Lab")

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
file = st.file_uploader("📂 Upload Movie CSV", type="csv")

if not file:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(file)

# -------------------------------------------------
# SAFE NORMALIZATION + BASE SCORE
# -------------------------------------------------
required_cols = {"rating", "votes", "revenue"}
df_score = None

if required_cols.issubset(df.columns):
    df_score = df.copy()

    for col in ["rating", "votes", "revenue"]:
        df_score[col] = pd.to_numeric(df_score[col], errors="coerce")

        min_val = df_score[col].min()
        max_val = df_score[col].max()

        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            df_score[col] = 0.0
        else:
            df_score[col] = (df_score[col] - min_val) / (max_val - min_val)

    df_score["success_score"] = (
        0.5 * df_score["rating"]
        + 0.3 * df_score["revenue"]
        + 0.2 * df_score["votes"]
    )

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploration",
    "🎯 Success Scoring",
    "🤖 Predictive Model",
    "📄 Report"
])

# -------------------------------------------------
# TAB 1 — EXPLORATION
# -------------------------------------------------
with tab1:
    st.subheader("Exploratory Analysis")
    st.dataframe(df.head())

    if "year" in df.columns:
        year_df = df["year"].dropna().astype(int)
        fig = px.line(
            year_df.value_counts().sort_index().reset_index(name="count"),
            x="index", y="count",
            title="Movies Released Per Year"
        )
        st.plotly_chart(fig, use_container_width=True)

    if {"genre", "rating"}.issubset(df.columns):
        fig = px.box(df, x="genre", y="rating", title="Rating Distribution by Genre")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 2 — SUCCESS SCORING
# -------------------------------------------------
with tab2:
    st.subheader("🎯 Movie Success Scoring")

    if df_score is None:
        st.warning("CSV must contain rating, votes, and revenue columns.")
    else:
        fig = px.histogram(
            df_score,
            x="success_score",
            title="Success Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            "Average Success Score",
            round(df_score["success_score"].mean(), 3)
        )

# -------------------------------------------------
# TAB 3 — PREDICTIVE MODEL
# -------------------------------------------------
with tab3:
    st.subheader("🤖 Predictive Model (With Metrics)")

    if df_score is None:
        st.warning("Success score unavailable.")
    else:
        model_df = df_score[
            ["rating", "votes", "revenue", "success_score"]
        ].dropna()

        if len(model_df) < 10:
            st.warning("Not enough clean data to train model.")
        else:
            X = model_df[["rating", "votes", "revenue"]]
            y = model_df["success_score"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            col1, col2 = st.columns(2)
            col1.metric("R² Score", f"{r2:.3f}")
            col2.metric("RMSE", f"{rmse:.3f}")

            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            })

            fig = px.bar(
                importance,
                x="Feature", y="Importance",
                title="Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 4 — REPORT (WITH VISUALS)
# -------------------------------------------------
with tab4:
    st.subheader("📄 Auto-Generated Report with Visualizations")

    if df_score is None:
        st.warning("Report unavailable.")
    else:
        insights = [
            f"Average success score: {df_score['success_score'].mean():.3f}",
            f"Maximum success score: {df_score['success_score'].max():.3f}",
            "Ratings contribute most strongly to movie success."
        ]

        for i in insights:
            st.write("-", i)

        if st.button("📥 Generate PDF Report"):
            # --- Chart 1 ---
            plt.figure()
            plt.hist(df_score["success_score"], bins=10)
            plt.title("Success Score Distribution")
            plt.xlabel("Success Score")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("success_dist.png")
            plt.close()

            # --- Chart 2 ---
            model_df = df_score[
                ["rating", "votes", "revenue", "success_score"]
            ].dropna()

            if len(model_df) >= 10:
                X = model_df[["rating", "votes", "revenue"]]
                y = model_df["success_score"]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                plt.figure()
                plt.bar(X.columns, model.feature_importances_)
                plt.title("Feature Importance")
                plt.tight_layout()
                plt.savefig("feature_importance.png")
                plt.close()

            # --- Build PDF ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, safe_text("Movie Intelligence Lab - Analysis Report"), ln=True)
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

            st.download_button(
                "⬇️ Download PDF Report",
                data=open("movie_analysis_report.pdf", "rb"),
                file_name="movie_analysis_report.pdf"
            )
