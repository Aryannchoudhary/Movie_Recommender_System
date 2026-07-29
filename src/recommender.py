# from src.content_based import ContentBasedRecommender
# from src.collaborative import CollaborativeRecommender
# from src.config import TOP_N_RECOMMENDATIONS


# class HybridRecommender:
#     """
#     Hybrid Movie Recommendation System

#     Supported Strategies:
#     ---------------------
#     - content
#     - collaborative
#     - genre
#     - popularity
#     """

#     def __init__(
#         self,
#         movies_df,
#         faiss_index,
#         indices,
#         cf_model,
#     ):

#         self.movies = movies_df

#         self.content = ContentBasedRecommender(
#             movies_df=movies_df,
#             faiss_index=faiss_index,
#             indices=indices,
#         )

#         self.collaborative = CollaborativeRecommender(
#             movies_df=movies_df,
#             cf_model=cf_model,
#         )

#     # --------------------------------------------------------
#     # Popular Movies
#     # --------------------------------------------------------

#     def popularity_recommend(self, top_n=TOP_N_RECOMMENDATIONS):

#         columns = [
#             "movie_idx",
#             "title",
#             "genres",
#             "overview",
#             "vote_average",
#             "vote_count",
#             "weighted_score",
#         ]

#         if "weighted_score" in self.movies.columns:

#             return (
#                 self.movies
#                 .sort_values(
#                     by="weighted_score",
#                     ascending=False,
#                 )[columns]
#                 .head(top_n)
#                 .reset_index(drop=True)
#             )

#         return (
#             self.movies
#             .sort_values(
#                 by="vote_average",
#                 ascending=False,
#             )
#             .head(top_n)
#             .reset_index(drop=True)
#         )

#     # --------------------------------------------------------
#     # Genre Recommendation
#     # --------------------------------------------------------

#     def genre_recommend(
#         self,
#         genres,
#         top_n=TOP_N_RECOMMENDATIONS,
#     ):

#         if isinstance(genres, str):
#             genres = [genres]

#         filtered = self.movies[
#             self.movies["genres"].str.contains(
#                 "|".join(genres),
#                 case=False,
#                 na=False,
#             )
#         ]

#         if filtered.empty:
#             return self.popularity_recommend(top_n)

#         sort_column = (
#             "weighted_score"
#             if "weighted_score" in filtered.columns
#             else "vote_average"
#         )

#         return (
#             filtered
#             .sort_values(
#                 by=sort_column,
#                 ascending=False,
#             )
#             .head(top_n)
#             .reset_index(drop=True)
#         )

#     # --------------------------------------------------------
#     # Main Recommendation Router
#     # --------------------------------------------------------

#     def recommend(
#         self,
#         strategy="content",
#         user_id=None,
#         movie_title=None,
#         genre=None,
#         exclude_movies=None,
#         top_n=TOP_N_RECOMMENDATIONS,
#     ):

#         if exclude_movies is None:
#             exclude_movies = []

#         strategy = strategy.lower()

#         # ----------------------------------------
#         # Collaborative Filtering
#         # ----------------------------------------

#         if strategy == "collaborative":

#             return {
#                 "strategy": "Collaborative Filtering",
#                 "results": self.collaborative.recommend(
#                     user_id=user_id,
#                     top_n=top_n,
#                     exclude_movies=exclude_movies,
#                 ),
#             }

#         # ----------------------------------------
#         # Content-Based
#         # ----------------------------------------

#         if strategy == "content":

#             return {
#                 "strategy": "Content-Based",
#                 "results": self.content.recommend(
#                     movie_title=movie_title,
#                     top_n=top_n,
#                 ),
#             }

#         # ----------------------------------------
#         # Genre-Based
#         # ----------------------------------------

#         if strategy == "genre":

#             return {
#                 "strategy": "Genre-Based",
#                 "results": self.genre_recommend(
#                     genres=genre,
#                     top_n=top_n,
#                 ),
#             }

#         # ----------------------------------------
#         # Popularity-Based
#         # ----------------------------------------

#         return {
#             "strategy": "Popularity-Based",
#             "results": self.popularity_recommend(
#                 top_n=top_n,
#             ),
#         }





import re

from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.config import TOP_N_RECOMMENDATIONS, MIN_INTERACTIONS_WARM


class HybridRecommender:
    """
    Hybrid Movie Recommendation System with AUTOMATIC warm/cold routing.

    Unlike a menu where the caller picks the algorithm, `recommend()`
    decides the strategy itself based on what's known about the user:

        user_id is warm (enough training interactions)  -> Collaborative Filtering
        user_id is cold/unknown + genre_profile given    -> Content-based, genre-anchored
        movie_title given (regardless of user_id)        -> Content-based, title-anchored
        nothing usable                                   -> Popularity baseline

    `force_strategy` is still available for manual testing/demoing each
    path individually, but it's opt-in -- the default behavior is automatic.
    """

    def __init__(
        self,
        movies_df,
        faiss_index,
        indices,
        cf_model,
        embed_model=None,
        train_interaction_counts=None,
        min_interactions_warm=MIN_INTERACTIONS_WARM,
    ):
        self.movies = movies_df

        self.content = ContentBasedRecommender(
            movies_df=movies_df,
            faiss_index=faiss_index,
            indices=indices,
            embed_model=embed_model,
        )

        self.collaborative = CollaborativeRecommender(
            movies_df=movies_df,
            cf_model=cf_model,
            train_interaction_counts=train_interaction_counts,
            min_interactions_warm=min_interactions_warm,
        )

    # --------------------------------------------------------
    # Popular Movies
    # --------------------------------------------------------

    def popularity_recommend(self, top_n=TOP_N_RECOMMENDATIONS):
        columns = ["movie_idx", "title", "genres", "overview", "vote_average", "vote_count", "weighted_score"]

        if "weighted_score" in self.movies.columns:
            return (
                self.movies.sort_values(by="weighted_score", ascending=False)[columns]
                .head(top_n)
                .reset_index(drop=True)
            )

        return self.movies.sort_values(by="vote_average", ascending=False).head(top_n).reset_index(drop=True)

    # --------------------------------------------------------
    # Genre Recommendation (simple substring filter fallback --
    # used when no embedding model is available for a semantic
    # genre_profile query)
    # --------------------------------------------------------

    def genre_recommend(self, genres, top_n=TOP_N_RECOMMENDATIONS):
        if isinstance(genres, str):
            genres = [genres]

        # Escape user input -- it's fed into a regex via str.contains, and
        # unescaped input can throw on special regex characters or match
        # unintended things.
        pattern = "|".join(re.escape(g) for g in genres if g)
        if not pattern:
            return self.popularity_recommend(top_n)

        filtered = self.movies[self.movies["genres"].str.contains(pattern, case=False, na=False, regex=True)]

        if filtered.empty:
            return self.popularity_recommend(top_n)

        sort_column = "weighted_score" if "weighted_score" in filtered.columns else "vote_average"
        return filtered.sort_values(by=sort_column, ascending=False).head(top_n).reset_index(drop=True)

    # --------------------------------------------------------
    # Automatic router
    # --------------------------------------------------------

    def recommend(
        self,
        user_id=None,
        movie_title=None,
        genre_profile=None,
        exclude_movies=None,
        top_n=TOP_N_RECOMMENDATIONS,
        force_strategy=None,
    ):
        """
        Parameters
        ----------
        user_id : int or None
            If given and warm, routes to Collaborative Filtering.
        movie_title : str or None
            If given, routes to content-based similarity anchored on this movie.
            Takes priority over genre_profile if both are given, since an
            explicit "movies like this" query is a stronger signal than a
            general taste profile.
        genre_profile : list[str] or None
            Genre preferences collected at signup -- the cold-start path for
            a brand-new user with no watch history and no specific movie in mind.
        force_strategy : str or None
            One of "collaborative", "content", "genre", "popularity".
            Bypasses automatic routing -- useful for demos/debugging, not
            meant to be the normal call path.
        """
        if exclude_movies is None:
            exclude_movies = []

        # ---- Manual override (opt-in, for demos/testing only) ----
        if force_strategy:
            strategy = force_strategy.lower()

            if strategy == "collaborative":
                return {"strategy": "Collaborative Filtering (forced)",
                        "results": self.collaborative.recommend(user_id, top_n, exclude_movies)}
            if strategy == "content":
                return {"strategy": "Content-Based (forced)",
                        "results": self.content.recommend(movie_title, top_n)}
            if strategy == "genre":
                return {"strategy": "Genre-Based (forced)",
                        "results": self.genre_recommend(genre_profile, top_n)}
            return {"strategy": "Popularity-Based (forced)",
                    "results": self.popularity_recommend(top_n)}

        # ---- Automatic routing ----

        if user_id is not None and self.collaborative.is_warm(user_id):
            return {
                "strategy": "Collaborative Filtering (warm user)",
                "results": self.collaborative.recommend(user_id, top_n, exclude_movies),
            }

        if movie_title:
            content_results = self.content.recommend(movie_title, top_n)
            if not content_results.empty:
                return {"strategy": "Content-Based (similar movies)", "results": content_results}
            # Title not found -> fall through to other signals rather than
            # returning nothing.

        if genre_profile:
            # Prefer the semantic embedding-based genre match if an embed
            # model was provided; otherwise fall back to substring filtering.
            semantic_results = self.content.recommend_from_genres(genre_profile, top_n)
            if not semantic_results.empty:
                return {"strategy": "Content-Based (genre profile, cold-start)", "results": semantic_results}
            return {
                "strategy": "Genre-Based (cold-start fallback)",
                "results": self.genre_recommend(genre_profile, top_n),
            }

        return {"strategy": "Popularity-Based (no signal available)", "results": self.popularity_recommend(top_n)}