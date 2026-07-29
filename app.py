


# import streamlit as st

# from src.loader import load_all_models
# from src.recommender import HybridRecommender
# from src.omdb import OMDbClient
# from src.utils import validate_user_id, dataframe_is_empty


# # ---------------------------------------------------------
# # Page Configuration
# # ---------------------------------------------------------

# st.set_page_config(
#     page_title="🎬 Movie Recommendation System",
#     page_icon="🎬",
#     layout="wide",
# )

# st.title("🎬 Movie Recommender")
# st.write(
#     "Tell us a movie you love, your favorite genres, or just browse what's "
#     "popular. We'll take it from there."
# )

# # ---------------------------------------------------------
# # Load Models
# # ---------------------------------------------------------

# with st.spinner("Loading recommendation models..."):
#     models = load_all_models()

# movies = models["movies"]
# faiss_index = models["faiss_index"]
# indices = models["indices"]
# cf_model = models["cf_model"]
# train_interaction_counts = models["train_interaction_counts"]
# embed_model = models["embed_model"]

# hybrid = HybridRecommender(
#     movies_df=movies,
#     faiss_index=faiss_index,
#     indices=indices,
#     cf_model=cf_model,
#     embed_model=embed_model,
#     train_interaction_counts=train_interaction_counts,
# )

# omdb = OMDbClient()


# # ---------------------------------------------------------
# # Display Movie Cards — grid layout instead of one full-width row per
# # movie, so several posters are visible at once without scrolling.
# # ---------------------------------------------------------

# POSTER_WIDTH = 140      # smaller poster so a card fits compactly
# CARDS_PER_ROW = 4       # tune this: 4-5 works well on a wide layout

# def display_movies(results):
#     if dataframe_is_empty(results):
#         st.warning("No recommendations found. Try a different movie, user ID, or genre.")
#         return

#     rows = results.reset_index(drop=True)

#     for start in range(0, len(rows), CARDS_PER_ROW):
#         chunk = rows.iloc[start:start + CARDS_PER_ROW]
#         cols = st.columns(CARDS_PER_ROW)

#         for col, (_, movie) in zip(cols, chunk.iterrows()):
#             with col:
#                 info = omdb.get_movie(movie["title"])

#                 if info and info.get("success") and info.get("poster"):
#                     st.image(info["poster"], width=POSTER_WIDTH)
#                 else:
#                     st.image(
#                         "https://via.placeholder.com/140x210?text=No+Poster",
#                         width=POSTER_WIDTH,
#                     )

#                 st.markdown(f"**{movie['title']}**")

#                 rating = info.get("imdb_rating") if info and info.get("success") else movie.get("vote_average")
#                 if rating and rating != "N/A":
#                     st.caption(f"⭐ {rating}")

#                 if "similarity_score" in movie:
#                     st.caption(f"Match: {float(movie['similarity_score']) * 100:.0f}%")
#                 elif "predicted_rating" in movie:
#                     st.caption(f"For you: {movie['predicted_rating']:.1f} / 5")

#                 with st.expander("Details"):
#                     if info and info.get("success"):
#                         st.write(f"📅 Year: {info.get('year', 'N/A')}")
#                         st.write(f"🎭 Genre: {info.get('genre', movie.get('genres', 'N/A'))}")
#                         plot = info.get("plot")
#                         if plot and plot != "N/A":
#                             st.write(plot)
#                     else:
#                         if "genres" in movie:
#                             st.write(f"🎭 Genre: {movie['genres']}")
#                         if "overview" in movie:
#                             st.write(movie["overview"])

#         st.write("")  # small vertical gap between rows


# # ---------------------------------------------------------
# # Sidebar — plain-language inputs. The router still works exactly the
# # same underneath; only the wording shown to the user changes here.
# # ---------------------------------------------------------

# st.sidebar.title("🎬 Find Your Next Movie")
# st.sidebar.caption("Fill in anything you know below — even just one thing helps.")

# user_id_input = st.sidebar.text_input(
#     "Returning user? Enter your ID",
#     value="",
#     placeholder="e.g. 42",
# )

# all_titles = sorted(
#     movies["title"].dropna().unique().tolist(),
#     key=lambda t: t.lower()
# ) if "title" in movies.columns else []

# MAX_SUGGESTIONS = 8  # cap suggestions -- never render the full ~100k-title list

# st.sidebar.markdown("**A movie you already love**")
# movie_search = st.sidebar.text_input(
#     "A movie you already love",
#     placeholder="Start typing a title... (e.g. Inception)",
#     label_visibility="collapsed",
# )

# movie_title = None
# movie_search_attempted = bool(movie_search.strip())

# if movie_search_attempted:
#     query = movie_search.strip().lower()

#     def _match_rank(title: str) -> int:
#         t = title.lower()
#         if t == query:
#             return 0        # exact match first
#         if t.startswith(query):
#             return 1        # then titles starting with the query
#         return 2             # then titles containing it anywhere

#     matches = sorted(
#         (t for t in all_titles if query in t.lower()),
#         key=lambda t: (_match_rank(t), t.lower()),
#     )[:MAX_SUGGESTIONS]

#     if matches:
#         movie_title = st.sidebar.selectbox(
#             "Pick the one you mean:",
#             options=["-- Select --"] + matches,
#         )
#         if movie_title == "-- Select --":
#             movie_title = None
#     else:
#         st.sidebar.warning(
#             f'No movie found matching "{movie_search}". '
#             "Try a different spelling, or leave this blank and use genres instead."
#         )
# else:
#     st.sidebar.caption("Type a few letters to see suggestions.")

# all_genres = sorted({
#     g.strip()
#     for genre_str in movies.get("genres", [])
#     for g in str(genre_str).split()
#     if g.strip()
# }) if "genres" in movies.columns else []

# genre_profile = st.sidebar.multiselect(
#     "A few genres you enjoy",
#     options=all_genres,
# )

# top_n = st.sidebar.slider("How many suggestions?", min_value=5, max_value=20, value=10)

# get_recs = st.sidebar.button("🎥 Find Movies For Me", type="primary", use_container_width=True)

# with st.sidebar.expander("🛠️ Developer / demo mode"):
#     st.caption("Manually pick a recommendation method instead of the automatic one.")
#     force_strategy = st.selectbox(
#         "Force method",
#         options=["None (automatic)", "collaborative", "content", "genre", "popularity"],
#     )
#     force_button = st.button("Run this method")


# # ---------------------------------------------------------
# # Translate internal strategy names into something a non-technical
# # visitor actually understands.
# # ---------------------------------------------------------

# def friendly_strategy_label(strategy: str, movie_title: str = None) -> str:
#     s = strategy.lower()

#     if "collaborative" in s:
#         return "🎯 Personalized picks based on people with similar taste to you"
#     if "content-based (similar" in s:
#         title_bit = f" '{movie_title}'" if movie_title else ""
#         return f"🎬 Because you liked{title_bit}"
#     if "genre" in s:
#         return "🍿 Matched to your favorite genres"
#     return "🔥 Popular right now"


# # ---------------------------------------------------------
# # Main panel
# # ---------------------------------------------------------

# def run_and_display(**kwargs):
#     response = hybrid.recommend(**kwargs)
#     label = friendly_strategy_label(response["strategy"], kwargs.get("movie_title"))
#     st.subheader(label)
#     display_movies(response["results"])


# if get_recs:
#     user_id = validate_user_id(user_id_input) if user_id_input.strip() else None

#     if movie_search_attempted and not movie_title:
#         st.info(
#             f'You searched for a movie but didn\'t pick one from the list — '
#             "showing recommendations based on what else you told us instead."
#         )

#     if not user_id_input.strip() and not movie_title and not genre_profile:
#         st.info("Didn't tell us anything specific? Here's what's popular right now 👇")

#     run_and_display(
#         user_id=user_id,
#         movie_title=movie_title if movie_title else None,
#         genre_profile=genre_profile if genre_profile else None,
#         top_n=top_n,
#     )

# elif force_button and force_strategy != "None (automatic)":
#     user_id = validate_user_id(user_id_input) if user_id_input.strip() else None

#     run_and_display(
#         user_id=user_id,
#         movie_title=movie_title if movie_title else None,
#         genre_profile=genre_profile if genre_profile else None,
#         top_n=top_n,
#         force_strategy=force_strategy,
#     )

# else:
#     st.write("👈 Tell us a bit about your taste and hit **Find Movies For Me**.")
























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

st.sidebar.markdown("**A movie you already love**")
movie_search = st.sidebar.text_input(
    "A movie you already love",
    placeholder="Type the exact title... (e.g. Inception)",
    label_visibility="collapsed",
)

movie_title = None
movie_search_attempted = bool(movie_search.strip())

if movie_search_attempted:
    query = movie_search.strip().lower()

    # Exact (case-insensitive) match only -- no suggestion list shown.
    exact_matches = [t for t in all_titles if t.lower() == query]

    if exact_matches:
        movie_title = exact_matches[0]
    else:
        st.sidebar.warning(
            f'No movie found matching "{movie_search}". '
            "Check the spelling, or leave this blank and use genres instead."
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