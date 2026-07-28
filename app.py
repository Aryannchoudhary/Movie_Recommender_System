import streamlit as st

from src.loader import load_all_models
from src.recommender import HybridRecommender
from src.omdb import OMDbClient


# Page Configuration


st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

st.title("Movie Recommender")
st.write(
    "Content-Based + Collaborative Filtering powered by FAISS, SentenceTransformer"
)

# Load Models

with st.spinner("Loading Recommendation Models..."):
    models = load_all_models()

movies = models["movies"]
faiss_index = models["faiss_index"]
indices = models["indices"]
cf_model = models["cf_model"]

# Initialize Recommender

hybrid = HybridRecommender(
    movies_df=movies,
    faiss_index=faiss_index,
    indices=indices,
    cf_model=cf_model,
)

omdb = OMDbClient()


# Display Movie Cards

def display_movies(results):

    if results is None or results.empty:
        st.warning("No recommendations found.")
        return

    for _, movie in results.iterrows():

        info = omdb.get_movie(movie["title"])

        col1, col2 = st.columns([1, 3])

        with col1:

            if (
                info
                and info.get("success")
                and info.get("poster")
                and info["poster"] != "N/A"
            ):
                st.image(info["poster"], width=180)
            else:
                st.image(
                    "https://via.placeholder.com/180x270?text=No+Poster",
                    width=180,
                )

        with col2:

            st.subheader(movie["title"])

            if "vote_average" in movie:
                st.write(f"⭐ TMDB Rating: {movie['vote_average']}")

            if info and info.get("success"):

                st.write(f"⭐ IMDb Rating: {info.get('imdb_rating', 'N/A')}")
                st.write(f"📅 Year: {info.get('year', 'N/A')}")
                st.write(f"🎭 Genre: {info.get('genre', movie.get('genres','N/A'))}")

                plot = info.get("plot")

                if plot and plot != "N/A":
                    st.write(plot)

            else:

                if "genres" in movie:
                    st.write(f"🎭 Genre: {movie['genres']}")

                if "overview" in movie:
                    st.write(movie["overview"])

            if "similarity_score" in movie:
                st.progress(float(movie["similarity_score"]))

        st.divider()


# Sidebar

st.sidebar.title("Recommendation Mode")

mode = st.sidebar.radio(
    "Choose Recommendation Type",
    (
        "Movie Title",
        "Genre",
        "Popular Movies",
    ),
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10,
)

# Movie Title

if mode == "Movie Title":

    movie_title = st.text_input("Enter Movie Title")

    if st.button("Recommend"):

        response = hybrid.recommend(
            strategy="content",
            movie_title=movie_title,
            top_n=top_n,
        )

        st.success(f"Strategy: {response['strategy']}")

        display_movies(response["results"])

# Genre

elif mode == "Genre":

    genre = st.text_input("Enter Genre")

    if st.button("Recommend"):

        response = hybrid.recommend(
            strategy="genre",
            genre=genre,
            top_n=top_n,
        )

        st.success(f"Strategy: {response['strategy']}")

        display_movies(response["results"])

# Collaborative

elif mode == "User ID":

    user_id = st.number_input(
        "User ID",
        min_value=0,
        step=1,
    )

    if st.button("Recommend"):

        response = hybrid.recommend(
            strategy="collaborative",
            user_id=int(user_id),
            top_n=top_n,
        )

        st.success(f"Strategy: {response['strategy']}")

        display_movies(response["results"])

# Popular Movies

else:

    if st.button("Show Popular Movies"):

        response = hybrid.recommend(
            strategy="popularity",
            top_n=top_n,
        )

        st.success(f"Strategy: {response['strategy']}")

        display_movies(response["results"])