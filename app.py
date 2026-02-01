import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🎬 Movie Analyzer", layout="wide")
sns.set_style("whitegrid")

st.title("🎬 Movie Analyzer")
st.write("Upload a movie CSV file and explore visual insights (No API required).")

# Sidebar
st.sidebar.header("⚙️ Filters")

uploaded_file = st.file_uploader("📂 Upload Movie CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Sidebar filters
    if "genre" in df.columns:
        genres = st.sidebar.multiselect(
            "Select Genre",
            df["genre"].dropna().unique(),
            default=df["genre"].dropna().unique()
        )
        df = df[df["genre"].isin(genres)]

    if "year" in df.columns:
        year_range = st.sidebar.slider(
            "Select Year Range",
            int(df["year"].min()),
            int(df["year"].max()),
            (int(df["year"].min()), int(df["year"].max()))
        )
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    # Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(df))
    if "rating" in df.columns:
        col2.metric("Avg Rating", round(df["rating"].mean(), 2))
    if "revenue" in df.columns:
        col3.metric("Total Revenue", round(df["revenue"].sum(), 2))

    # Rating Distribution
    if "rating" in df.columns:
        st.subheader("⭐ Rating Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["rating"], bins=10, kde=True, ax=ax)
        st.pyplot(fig)

    # Top Rated Movies
    if "rating" in df.columns and "title" in df.columns:
        st.subheader("🏆 Top 10 Rated Movies")
        top_movies = df.sort_values(by="rating", ascending=False).head(10)
        st.table(top_movies[["title", "rating"]])

    # Genre-wise Average Rating
    if "genre" in df.columns and "rating" in df.columns:
        st.subheader("🎭 Genre-wise Average Rating")
        genre_avg = df.groupby("genre")["rating"].mean().sort_values(ascending=False)
        st.bar_chart(genre_avg)

    # Revenue vs Rating
    if "revenue" in df.columns and "rating" in df.columns:
        st.subheader("💰 Revenue vs Rating")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="rating", y="revenue", ax=ax)
        st.pyplot(fig)

    # Votes vs Rating (Bubble Chart)
    if {"votes", "rating"}.issubset(df.columns):
        st.subheader("🗳 Votes vs Rating")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x="rating",
            y="votes",
            size="votes",
            sizes=(20, 300),
            alpha=0.6,
            ax=ax
        )
        st.pyplot(fig)

    # Movies per Year
    if "year" in df.columns:
        st.subheader("📅 Movies Released per Year")
        year_count = df["year"].value_counts().sort_index()
        st.line_chart(year_count)

    # Correlation Heatmap
    numeric_cols = df.select_dtypes(include=["int64", "float64"])
    if len(numeric_cols.columns) > 1:
        st.subheader("🔗 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

else:
    st.info("⬆️ Upload a CSV file to start visualization.")
