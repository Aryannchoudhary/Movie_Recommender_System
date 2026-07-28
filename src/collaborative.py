import pandas as pd


class CollaborativeRecommender:
    """
    Collaborative Filtering Recommender using Surprise SVD.
    """

    def __init__(self, movies_df, cf_model):
        """
        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataset.

        cf_model : Surprise SVD model
            Trained collaborative filtering model.
        """
        self.movies = movies_df
        self.cf_model = cf_model

    def recommend(
        self,
        user_id,
        top_n=10,
        exclude_movies=None,
    ):
        """
        Recommend movies for a given user.

        Parameters
        ----------
        user_id : int
            User ID.

        top_n : int
            Number of recommendations.

        exclude_movies : list
            Movies already watched by the user.

        Returns
        -------
        pd.DataFrame
        """

        if exclude_movies is None:
            exclude_movies = []

        # Candidate movies
        candidates = self.movies[
            ~self.movies["movie_idx"].isin(exclude_movies)
        ].copy()

        # Predict ratings
        candidates["predicted_rating"] = candidates["movie_idx"].apply(
            lambda movie_id: self.cf_model.predict(user_id, movie_id).est
        )

        # Sort by predicted rating
        recommendations = (
            candidates
            .sort_values(
                by="predicted_rating",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        columns = [
            "movie_idx",
            "title",
            "genres",
            "overview",
            "vote_average",
            "predicted_rating",
        ]

        return recommendations[columns]