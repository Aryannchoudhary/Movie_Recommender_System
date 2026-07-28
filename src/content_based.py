import numpy as np
import pandas as pd


class ContentBasedRecommender:
    """
    Content-Based Movie Recommendation using FAISS.
    """

    def __init__(self, movies_df, faiss_index, indices):
        self.movies = movies_df
        self.index = faiss_index
        self.indices = indices

    # ---------------------------------------------------------
    # Check whether movie exists
    # ---------------------------------------------------------

    def movie_exists(self, title: str) -> bool:
        """
        Check if the movie exists in the dataset.
        """

        if title is None:
            return False

        if not isinstance(title, str):
            return False

        title = title.strip()

        if title == "":
            return False

        return title.lower() in self.indices

    

    # ---------------------------------------------------------
    # Get dataframe index
    # ---------------------------------------------------------

    def get_movie_index(self, title: str):
        """
        Get dataframe index of the movie.
        """

        return self.indices.get(title.lower())

    # ---------------------------------------------------------
    # Recommend Similar Movies
    # ---------------------------------------------------------

    def recommend(
        self,
        movie_title: str,
        top_n: int = 10,
    ):
        """
        Recommend movies similar to the selected movie.
        """

        # Empty input
        if not self.movie_exists(movie_title):
            return pd.DataFrame()

        movie_idx = self.get_movie_index(movie_title)

        if movie_idx is None:
            return pd.DataFrame()

        try:
            # Get embedding from FAISS
            query_vector = self.index.reconstruct(movie_idx).reshape(1, -1)

            # Search similar vectors
            distances, indices = self.index.search(
                query_vector,
                top_n + 1,
            )

        except Exception as e:
            print(f"FAISS Error: {e}")
            return pd.DataFrame()

        similar_indices = indices.flatten()[1:]
        similarity_scores = distances.flatten()[1:]

        columns = [
            "movie_idx",
            "title",
            "genres",
            "overview",
            "vote_average"
        ]

        recommendations = (
            self.movies.iloc[similar_indices][columns]
            .copy()
            .reset_index(drop=True)
        )

        recommendations["similarity_score"] = similarity_scores

        return (
            recommendations
            .sort_values(
                by="similarity_score",
                ascending=False,
            )
            .reset_index(drop=True)
        )