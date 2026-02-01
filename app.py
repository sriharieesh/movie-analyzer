import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🎬 Movie Analyzer", layout="wide")

st.title("🎬 Movie Analyzer (CSV Upload)")
st.write("Upload a movie dataset and explore insights — **no API key required**.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your movie CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Basic info
    st.subheader("ℹ️ Dataset Info")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Movies", len(df))

    with col2:
        st.metric("Total Columns", len(df.columns))

    # Numeric analysis
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_cols:
        st.subheader("📈 Statistical Summary")
        st.dataframe(df[numeric_cols].describe())

    # Rating analysis
    if "rating" in df.columns:
        st.subheader("⭐ Ratings Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["rating"], bins=10, kde=True, ax=ax)
        st.pyplot(fig)

    # Genre analysis
    if "genre" in df.columns:
        st.subheader("🎭 Movies by Genre")
        genre_count = df["genre"].value_counts().head(10)
        st.bar_chart(genre_count)

    # Revenue vs Rating
    if "revenue" in df.columns and "rating" in df.columns:
        st.subheader("💰 Revenue vs Rating")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=df["rating"],
            y=df["revenue"],
            ax=ax
        )
        ax.set_xlabel("Rating")
        ax.set_ylabel("Revenue")
        st.pyplot(fig)

    # Year-wise analysis
    if "year" in df.columns:
        st.subheader("📅 Movies by Year")
        year_count = df["year"].value_counts().sort_index()
        st.line_chart(year_count)

else:
    st.info("👆 Upload a CSV file to start analyzing movies.")
