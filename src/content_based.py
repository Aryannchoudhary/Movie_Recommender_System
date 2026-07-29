# import numpy as np
# import pandas as pd


# class ContentBasedRecommender:
#     """
#     Content-Based Movie Recommendation using FAISS.
#     """

#     def __init__(self, movies_df, faiss_index, indices):
#         self.movies = movies_df
#         self.index = faiss_index
#         self.indices = indices

#     # ---------------------------------------------------------
#     # Check whether movie exists
#     # ---------------------------------------------------------

#     def movie_exists(self, title: str) -> bool:
#         """
#         Check if the movie exists in the dataset.
#         """

#         if title is None:
#             return False

#         if not isinstance(title, str):
#             return False

#         title = title.strip()

#         if title == "":
#             return False

#         return title.lower() in self.indices

    

#     # ---------------------------------------------------------
#     # Get dataframe index
#     # ---------------------------------------------------------

#     def get_movie_index(self, title: str):
#         """
#         Get dataframe index of the movie.
#         """

#         return self.indices.get(title.lower())

#     # ---------------------------------------------------------
#     # Recommend Similar Movies
#     # ---------------------------------------------------------

#     def recommend(
#         self,
#         movie_title: str,
#         top_n: int = 10,
#     ):
#         """
#         Recommend movies similar to the selected movie.
#         """

#         # Empty input
#         if not self.movie_exists(movie_title):
#             return pd.DataFrame()

#         movie_idx = self.get_movie_index(movie_title)

#         if movie_idx is None:
#             return pd.DataFrame()

#         try:
#             # Get embedding from FAISS
#             query_vector = self.index.reconstruct(movie_idx).reshape(1, -1)

#             # Search similar vectors
#             distances, indices = self.index.search(
#                 query_vector,
#                 top_n + 1,
#             )

#         except Exception as e:
#             print(f"FAISS Error: {e}")
#             return pd.DataFrame()

#         similar_indices = indices.flatten()[1:]
#         similarity_scores = distances.flatten()[1:]

#         columns = [
#             "movie_idx",
#             "title",
#             "genres",
#             "overview",
#             "vote_average"
#         ]

#         recommendations = (
#             self.movies.iloc[similar_indices][columns]
#             .copy()
#             .reset_index(drop=True)
#         )

#         recommendations["similarity_score"] = similarity_scores

#         return (
#             recommendations
#             .sort_values(
#                 by="similarity_score",
#                 ascending=False,
#             )
#             .reset_index(drop=True)
#         )






import numpy as np
import pandas as pd


class ContentBasedRecommender:
    """
    Content-Based Movie Recommendation using FAISS.

    Supports two anchor types:
    - movie_title: recommend movies similar to an existing movie
      (uses the movie's already-stored embedding via index.reconstruct,
      no re-encoding needed)
    - genre_profile: recommend movies for a brand-new user with no
      interaction history at all, based on a list of preferred genres
      collected at signup (requires an embed_model to encode the
      synthetic genre-preference text into the same vector space)
    """

    def __init__(self, movies_df, faiss_index, indices, embed_model=None):
        self.movies = movies_df
        self.index = faiss_index
        self.indices = indices
        self.embed_model = embed_model

    # ---------------------------------------------------------
    # Check whether movie exists
    # ---------------------------------------------------------

    def movie_exists(self, title: str) -> bool:
        if not isinstance(title, str):
            return False
        title = title.strip()
        if title == "":
            return False
        return title.lower() in self.indices

    def get_movie_index(self, title: str):
        return self.indices.get(title.lower())

    # ---------------------------------------------------------
    # Shared: turn FAISS search results into a results dataframe
    # ---------------------------------------------------------

    def _build_results(self, query_vector, top_n, exclude_idx=None):
        exclude_idx = set(exclude_idx or [])

        # over-fetch a bit so we still have top_n left after exclusions
        distances, indices = self.index.search(query_vector, top_n + len(exclude_idx) + 1)

        similar_indices = indices.flatten()
        similarity_scores = distances.flatten()

        columns = ["movie_idx", "title", "genres", "overview", "vote_average"]
        rows = []
        for idx, score in zip(similar_indices, similarity_scores):
            if idx == -1 or idx in exclude_idx:
                continue
            rows.append((idx, score))
            if len(rows) >= top_n:
                break

        if not rows:
            return pd.DataFrame()

        row_indices = [r[0] for r in rows]
        scores = [r[1] for r in rows]

        recommendations = self.movies.iloc[row_indices][columns].copy().reset_index(drop=True)
        # Clip to [0, 1] -- inner-product scores on normalized embeddings are
        # usually in this range but aren't strictly guaranteed to be, and
        # st.progress() raises an exception outside [0, 1].
        recommendations["similarity_score"] = np.clip(scores, 0.0, 1.0)

        return recommendations.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------
    # Recommend similar to an existing movie
    # ---------------------------------------------------------

    def recommend(self, movie_title: str, top_n: int = 10):
        if not self.movie_exists(movie_title):
            return pd.DataFrame()

        movie_idx = self.get_movie_index(movie_title)
        if movie_idx is None:
            return pd.DataFrame()

        try:
            query_vector = self.index.reconstruct(int(movie_idx)).reshape(1, -1)
        except Exception as e:
            print(f"FAISS Error: {e}")
            return pd.DataFrame()

        return self._build_results(query_vector, top_n, exclude_idx={movie_idx})

    # ---------------------------------------------------------
    # Recommend from a genre profile (cold-start / new-user path)
    # ---------------------------------------------------------

    def recommend_from_genres(self, genre_profile, top_n: int = 10):
        """
        For a brand-new user with no watch history: build a pseudo-query
        from their stated genre preferences (collected at signup) and find
        the closest movies in the same embedding space.
        """
        if not genre_profile:
            return pd.DataFrame()

        if self.embed_model is None:
            # No embedding model available at runtime -> can't do a semantic
            # genre query; caller should fall back to simple genre filtering.
            return pd.DataFrame()

        pseudo_text = " ".join(genre_profile)
        query_vector = self.embed_model.encode(
            [pseudo_text], normalize_embeddings=True
        ).astype("float32")

        return self._build_results(query_vector, top_n)