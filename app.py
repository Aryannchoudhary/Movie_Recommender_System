# import streamlit as st

# from src.loader import load_all_models
# from src.recommender import HybridRecommender
# from src.omdb import OMDbClient


# # Page Configuration


# st.set_page_config(
#     page_title="🎬 Movie Recommendation System",
#     page_icon="🎬",
#     layout="wide",
# )

# st.title("Movie Recommender")
# st.write(
#     "Content-Based + Collaborative Filtering powered by FAISS, SentenceTransformer"
# )

# # Load Models

# with st.spinner("Loading Recommendation Models..."):
#     models = load_all_models()

# movies = models["movies"]
# faiss_index = models["faiss_index"]
# indices = models["indices"]
# cf_model = models["cf_model"]

# # Initialize Recommender

# hybrid = HybridRecommender(
#     movies_df=movies,
#     faiss_index=faiss_index,
#     indices=indices,
#     cf_model=cf_model,
# )

# omdb = OMDbClient()


# # Display Movie Cards

# def display_movies(results):

#     if results is None or results.empty:
#         st.warning("No recommendations found.")
#         return

#     for _, movie in results.iterrows():

#         info = omdb.get_movie(movie["title"])

#         col1, col2 = st.columns([1, 3])

#         with col1:

#             if (
#                 info
#                 and info.get("success")
#                 and info.get("poster")
#                 and info["poster"] != "N/A"
#             ):
#                 st.image(info["poster"], width=180)
#             else:
#                 st.image(
#                     "https://via.placeholder.com/180x270?text=No+Poster",
#                     width=180,
#                 )

#         with col2:

#             st.subheader(movie["title"])

#             if "vote_average" in movie:
#                 st.write(f"⭐ TMDB Rating: {movie['vote_average']}")

#             if info and info.get("success"):

#                 st.write(f"⭐ IMDb Rating: {info.get('imdb_rating', 'N/A')}")
#                 st.write(f"📅 Year: {info.get('year', 'N/A')}")
#                 st.write(f"🎭 Genre: {info.get('genre', movie.get('genres','N/A'))}")

#                 plot = info.get("plot")

#                 if plot and plot != "N/A":
#                     st.write(plot)

#             else:

#                 if "genres" in movie:
#                     st.write(f"🎭 Genre: {movie['genres']}")

#                 if "overview" in movie:
#                     st.write(movie["overview"])

#             if "similarity_score" in movie:
#                 st.progress(float(movie["similarity_score"]))

#         st.divider()


# # Sidebar

# st.sidebar.title("Recommendation Mode")

# mode = st.sidebar.radio(
#     "Choose Recommendation Type",
#     (
#         "Movie Title",
#         "Genre",
#         "Popular Movies",
#     ),
# )

# top_n = st.sidebar.slider(
#     "Number of Recommendations",
#     min_value=5,
#     max_value=20,
#     value=10,
# )

# # Movie Title

# if mode == "Movie Title":

#     movie_title = st.text_input("Enter Movie Title")

#     if st.button("Recommend"):

#         response = hybrid.recommend(
#             strategy="content",
#             movie_title=movie_title,
#             top_n=top_n,
#         )

#         st.success(f"Strategy: {response['strategy']}")

#         display_movies(response["results"])

# # Genre

# elif mode == "Genre":

#     genre = st.text_input("Enter Genre")

#     if st.button("Recommend"):

#         response = hybrid.recommend(
#             strategy="genre",
#             genre=genre,
#             top_n=top_n,
#         )

#         st.success(f"Strategy: {response['strategy']}")

#         display_movies(response["results"])

# # Collaborative

# elif mode == "User ID":

#     user_id = st.number_input(
#         "User ID",
#         min_value=0,
#         step=1,
#     )

#     if st.button("Recommend"):

#         response = hybrid.recommend(
#             strategy="collaborative",
#             user_id=int(user_id),
#             top_n=top_n,
#         )

#         st.success(f"Strategy: {response['strategy']}")

#         display_movies(response["results"])

# # Popular Movies

# else:

#     if st.button("Show Popular Movies"):

#         response = hybrid.recommend(
#             strategy="popularity",
#             top_n=top_n,
#         )

#         st.success(f"Strategy: {response['strategy']}")

#         display_movies(response["results"])



import streamlit as st

from src.loader import load_all_models
from src.recommender import HybridRecommender
from src.omdb import OMDbClient
from src.utils import validate_user_id, dataframe_is_empty


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Recommender")
st.write(
    "Tell us a movie you love, your favorite genres, or just browse what's "
    "popular. We'll take it from there."
)

# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------

with st.spinner("Loading recommendation models..."):
    models = load_all_models()

movies = models["movies"]
faiss_index = models["faiss_index"]
indices = models["indices"]
cf_model = models["cf_model"]
train_interaction_counts = models["train_interaction_counts"]
embed_model = models["embed_model"]

hybrid = HybridRecommender(
    movies_df=movies,
    faiss_index=faiss_index,
    indices=indices,
    cf_model=cf_model,
    embed_model=embed_model,
    train_interaction_counts=train_interaction_counts,
)

omdb = OMDbClient()


# ---------------------------------------------------------
# Display Movie Cards — grid layout instead of one full-width row per
# movie, so several posters are visible at once without scrolling.
# ---------------------------------------------------------

POSTER_WIDTH = 140      # smaller poster so a card fits compactly
CARDS_PER_ROW = 4       # tune this: 4-5 works well on a wide layout

def display_movies(results):
    if dataframe_is_empty(results):
        st.warning("No recommendations found. Try a different movie, user ID, or genre.")
        return

    rows = results.reset_index(drop=True)

    for start in range(0, len(rows), CARDS_PER_ROW):
        chunk = rows.iloc[start:start + CARDS_PER_ROW]
        cols = st.columns(CARDS_PER_ROW)

        for col, (_, movie) in zip(cols, chunk.iterrows()):
            with col:
                info = omdb.get_movie(movie["title"])

                if info and info.get("success") and info.get("poster"):
                    st.image(info["poster"], width=POSTER_WIDTH)
                else:
                    st.image(
                        "https://via.placeholder.com/140x210?text=No+Poster",
                        width=POSTER_WIDTH,
                    )

                st.markdown(f"**{movie['title']}**")

                rating = info.get("imdb_rating") if info and info.get("success") else movie.get("vote_average")
                if rating and rating != "N/A":
                    st.caption(f"⭐ {rating}")

                if "similarity_score" in movie:
                    st.caption(f"Match: {float(movie['similarity_score']) * 100:.0f}%")
                elif "predicted_rating" in movie:
                    st.caption(f"For you: {movie['predicted_rating']:.1f} / 5")

                with st.expander("Details"):
                    if info and info.get("success"):
                        st.write(f"📅 Year: {info.get('year', 'N/A')}")
                        st.write(f"🎭 Genre: {info.get('genre', movie.get('genres', 'N/A'))}")
                        plot = info.get("plot")
                        if plot and plot != "N/A":
                            st.write(plot)
                    else:
                        if "genres" in movie:
                            st.write(f"🎭 Genre: {movie['genres']}")
                        if "overview" in movie:
                            st.write(movie["overview"])

        st.write("")  # small vertical gap between rows


# ---------------------------------------------------------
# Sidebar — plain-language inputs. The router still works exactly the
# same underneath; only the wording shown to the user changes here.
# ---------------------------------------------------------

st.sidebar.title("🎬 Find Your Next Movie")
st.sidebar.caption("Fill in anything you know below — even just one thing helps.")

user_id_input = st.sidebar.text_input(
    "Returning user? Enter your ID",
    value="",
    placeholder="e.g. 42",
)

all_titles = sorted(
    movies["title"].dropna().unique().tolist(),
    key=lambda t: t.lower()
) if "title" in movies.columns else []

movie_title = st.sidebar.selectbox(
    "A movie you already love",
    options=[""] + all_titles,
    index=0,
    placeholder="Start typing a title...",
)

all_genres = sorted({
    g.strip()
    for genre_str in movies.get("genres", [])
    for g in str(genre_str).split()
    if g.strip()
}) if "genres" in movies.columns else []

genre_profile = st.sidebar.multiselect(
    "A few genres you enjoy",
    options=all_genres,
)

top_n = st.sidebar.slider("How many suggestions?", min_value=5, max_value=20, value=10)

get_recs = st.sidebar.button("🎥 Find Movies For Me", type="primary", use_container_width=True)

with st.sidebar.expander("🛠️ Developer / demo mode"):
    st.caption("Manually pick a recommendation method instead of the automatic one.")
    force_strategy = st.selectbox(
        "Force method",
        options=["None (automatic)", "collaborative", "content", "genre", "popularity"],
    )
    force_button = st.button("Run this method")


# ---------------------------------------------------------
# Translate internal strategy names into something a non-technical
# visitor actually understands.
# ---------------------------------------------------------

def friendly_strategy_label(strategy: str, movie_title: str = None) -> str:
    s = strategy.lower()

    if "collaborative" in s:
        return "🎯 Personalized picks based on people with similar taste to you"
    if "content-based (similar" in s:
        title_bit = f" '{movie_title}'" if movie_title else ""
        return f"🎬 Because you liked{title_bit}"
    if "genre" in s:
        return "🍿 Matched to your favorite genres"
    return "🔥 Popular right now"


# ---------------------------------------------------------
# Main panel
# ---------------------------------------------------------

def run_and_display(**kwargs):
    response = hybrid.recommend(**kwargs)
    label = friendly_strategy_label(response["strategy"], kwargs.get("movie_title"))
    st.subheader(label)
    display_movies(response["results"])


if get_recs:
    user_id = validate_user_id(user_id_input) if user_id_input.strip() else None

    if not user_id_input.strip() and not movie_title and not genre_profile:
        st.info("Didn't tell us anything specific? Here's what's popular right now 👇")

    run_and_display(
        user_id=user_id,
        movie_title=movie_title if movie_title else None,
        genre_profile=genre_profile if genre_profile else None,
        top_n=top_n,
    )

elif force_button and force_strategy != "None (automatic)":
    user_id = validate_user_id(user_id_input) if user_id_input.strip() else None

    run_and_display(
        user_id=user_id,
        movie_title=movie_title if movie_title else None,
        genre_profile=genre_profile if genre_profile else None,
        top_n=top_n,
        force_strategy=force_strategy,
    )

else:
    st.write("👈 Tell us a bit about your taste and hit **Find Movies For Me**.")