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

    # default weights
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
# TAB 2 — USER-CONTROLLED SUCCESS SCORING
# -------------------------------------------------
with tab2:
    st.subheader("🎯 Movie Success Scoring (User Controlled)")

    if df_score is None:
        st.warning("CSV must contain rating, votes, and revenue columns.")
    else:
        st.markdown("### Adjust Feature Weights")

        w_rating = st.slider("Rating Weight", 0.0, 1.0, 0.5, 0.05)
        w_revenue = st.slider(
            "Revenue Weight", 0.0, round(1.0 - w_rating, 2), 0.3, 0.05
        )
        w_votes = round(1.0 - (w_rating + w_revenue), 2)

        st.info(f"Votes Weight auto-set to **{w_votes}**")

        df_score["success_score"] = (
            w_rating * df_score["rating"]
            + w_revenue * df_score["revenue"]
            + w_votes * df_score["votes"]
        )

        st.metric(
            "Average Success Score",
            round(df_score["success_score"].mean(), 3)
        )

        fig = px.histogram(
            df_score,
            x="success_score",
            title="Success Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 3 — ML MODEL + METRICS
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

            # ✅ VERSION-SAFE METRICS (NO squared=FALSE)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))

            col1, col2 = st.columns(2)
            col1.metric("R² Score", f"{r2:.3f}")
            col2.metric("RMSE", f"{rmse:.3f}")

            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig = px.bar(
                importance,
                x="Feature",
                y="Importance",
                title="Feature Importance on Success Prediction"
            )
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# TAB 4 — REPORT
# -------------------------------------------------
with tab4:
    st.subheader("📄 Auto-Generated Report with Visualizations")

    if df_score is None:
        st.warning("Report unavailable.")
    else:
        # ---------- INSIGHTS ----------
        insights = [
            f"Average success score: {df_score['success_score'].mean():.3f}",
            f"Maximum success score: {df_score['success_score'].max():.3f}",
            "Ratings contribute most strongly to movie success."
        ]

        for i in insights:
            st.write("•", i)

        if st.button("📥 Generate PDF Report"):
            # ---------- MATPLOTLIB FIG 1 ----------
            plt.figure()
            plt.hist(df_score["success_score"], bins=10)
            plt.title("Success Score Distribution")
            plt.xlabel("Success Score")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("success_dist.png")
            plt.close()

            # ---------- MATPLOTLIB FIG 2 ----------
            model_df = df_score[
                ["rating", "votes", "revenue", "success_score"]
            ].dropna()

            if len(model_df) >= 10:
                X = model_df[["rating", "votes", "revenue"]]
                y = model_df["success_score"]

                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
                model.fit(X, y)

                importances = model.feature_importances_

                plt.figure()
                plt.bar(X.columns, importances)
                plt.title("Feature Importance")
                plt.xlabel("Feature")
                plt.ylabel("Importance")
                plt.tight_layout()
                plt.savefig("feature_importance.png")
                plt.close()

            # ---------- BUILD PDF ----------
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Movie Intelligence Lab – Analysis Report", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", size=11)
            for i in insights:
                pdf.multi_cell(0, 8, f"• {i}")
            pdf.ln(5)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Success Score Distribution", ln=True)
            pdf.image("success_dist.png", w=170)
            pdf.ln(5)

            if len(model_df) >= 10:
                pdf.cell(0, 10, "Feature Importance", ln=True)
                pdf.image("feature_importance.png", w=170)

            pdf.output("movie_analysis_report.pdf")

            st.download_button(
                "⬇️ Download PDF Report",
                data=open("movie_analysis_report.pdf", "rb"),
                file_name="movie_analysis_report.pdf"
            )
