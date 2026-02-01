import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fpdf import FPDF

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="🎬 Movie Analyzer Pro", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.metric { background: #161b22; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Analyzer Pro Dashboard")

# ---------------- UPLOAD ----------------
file = st.file_uploader("📂 Upload Movie CSV", type="csv")

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(df))

    if "rating" in df.columns:
        col2.metric("Average Rating", round(df["rating"].mean(), 2))

    if "revenue" in df.columns:
        col3.metric("Total Revenue", round(df["revenue"].sum(), 2))

    # ---------------- INTERACTIVE PLOTS ----------------
    st.subheader("📊 Interactive Visualizations")

    if "rating" in df.columns:
        fig = px.histogram(df, x="rating", nbins=10, title="Rating Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if {"genre", "rating"}.issubset(df.columns):
        fig = px.bar(
            df.groupby("genre")["rating"].mean().reset_index(),
            x="genre", y="rating",
            title="Genre vs Average Rating"
        )
        st.plotly_chart(fig, use_container_width=True)

    if {"rating", "revenue"}.issubset(df.columns):
        fig = px.scatter(
            df, x="rating", y="revenue",
            size="revenue", hover_name="title",
            title="Revenue vs Rating"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- ML PREDICTION ----------------
    st.subheader("🤖 Movie Success Prediction")

    if {"rating", "votes", "revenue"}.issubset(df.columns):
        df_ml = df.dropna(subset=["rating", "votes", "revenue"])

        df_ml["success"] = (df_ml["rating"] >= 7).astype(int)

        X = df_ml[["rating", "votes", "revenue"]]
        y = df_ml["success"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"🎯 Model Accuracy: {round(acc * 100, 2)}%")

    # ---------------- AI-LIKE EDA SUMMARY ----------------
    st.subheader("🧠 Automated EDA Insights")

    insights = []

    if "rating" in df.columns:
        insights.append(f"• Average movie rating is {df['rating'].mean():.2f}")

    if "revenue" in df.columns:
        insights.append(
            f"• Highest revenue movie earned {df['revenue'].max():.2f}"
        )

    if "year" in df.columns:
        peak_year = df["year"].value_counts().idxmax()
        insights.append(f"• Most movies were released in {peak_year}")

    for i in insights:
        st.write(i)

    # ---------------- PDF REPORT ----------------
    st.subheader("📄 Generate PDF Report")

    if st.button("📥 Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Movie Analyzer Report", ln=True, align="C")
        pdf.ln(10)

        for line in insights:
            pdf.multi_cell(0, 8, line)

        pdf.output("movie_report.pdf")
        st.success("PDF generated successfully!")
        st.download_button(
            "⬇️ Download Report",
            data=open("movie_report.pdf", "rb"),
            file_name="movie_report.pdf"
        )

else:
    st.info("⬆️ Upload a CSV file to unlock the dashboard.")
