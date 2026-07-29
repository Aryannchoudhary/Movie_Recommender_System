# import pandas as pd


# class CollaborativeRecommender:
#     """
#     Collaborative Filtering Recommender using Surprise SVD.
#     """

#     def __init__(self, movies_df, cf_model):
#         """
#         Parameters
#         ----------
#         movies_df : pd.DataFrame
#             Movie dataset.

#         cf_model : Surprise SVD model
#             Trained collaborative filtering model.
#         """
#         self.movies = movies_df
#         self.cf_model = cf_model

#     def recommend(
#         self,
#         user_id,
#         top_n=10,
#         exclude_movies=None,
#     ):
#         """
#         Recommend movies for a given user.

#         Parameters
#         ----------
#         user_id : int
#             User ID.

#         top_n : int
#             Number of recommendations.

#         exclude_movies : list
#             Movies already watched by the user.

#         Returns
#         -------
#         pd.DataFrame
#         """

#         if exclude_movies is None:
#             exclude_movies = []

#         # Candidate movies
#         candidates = self.movies[
#             ~self.movies["movie_idx"].isin(exclude_movies)
#         ].copy()

#         # Predict ratings
#         candidates["predicted_rating"] = candidates["movie_idx"].apply(
#             lambda movie_id: self.cf_model.predict(user_id, movie_id).est
#         )

#         # Sort by predicted rating
#         recommendations = (
#             candidates
#             .sort_values(
#                 by="predicted_rating",
#                 ascending=False
#             )
#             .head(top_n)
#             .reset_index(drop=True)
#         )

#         columns = [
#             "movie_idx",
#             "title",
#             "genres",
#             "overview",
#             "vote_average",
#             "predicted_rating",
#         ]

#         return recommendations[columns]




import numpy as np
import pandas as pd


class CollaborativeRecommender:
    """
    Collaborative Filtering Recommender using Surprise SVD.

    Includes:
    - is_warm(user_id): whether the user has enough training interactions
      for CF predictions to be meaningfully personalized (vs. an untrained
      user falling back to global/item-bias-only scores).
    - Vectorized scoring: scores every candidate movie via a single matrix
      operation instead of looping cf_model.predict() per movie, which
      matters once the candidate pool gets into the tens/hundreds of
      thousands of movies.
    """

    def __init__(self, movies_df, cf_model, train_interaction_counts=None,
                 min_interactions_warm=5):
        self.movies = movies_df
        self.cf_model = cf_model
        self.train_interaction_counts = train_interaction_counts or {}
        self.min_interactions_warm = min_interactions_warm

    # ---------------------------------------------------------
    # Warm / cold check
    # ---------------------------------------------------------

    def is_warm(self, user_id) -> bool:
        """
        A user is 'warm' if they had enough interactions in the training
        data for CF to have learned a real (non-default) user vector.
        Unknown/new users are always cold.
        """
        if user_id is None:
            return False

        try:
            # Confirms the user actually exists in the trained model at all
            self.cf_model.trainset.to_inner_uid(user_id)
        except ValueError:
            return False

        return self.train_interaction_counts.get(user_id, 0) >= self.min_interactions_warm

    # ---------------------------------------------------------
    # Vectorized scoring
    # ---------------------------------------------------------

    def _score_all_candidates(self, user_id, candidate_movie_ids):
        """
        Scores every candidate in one vectorized pass using the SVD model's
        raw factors, instead of calling cf_model.predict() per movie.
        """
        trainset = self.cf_model.trainset
        global_mean = trainset.global_mean

        try:
            inner_uid = trainset.to_inner_uid(user_id)
            user_known = True
        except ValueError:
            user_known = False

        # Map candidate raw movie ids -> inner item ids the model actually knows
        inner_ids, raw_ids_known = [], []
        for raw_id in candidate_movie_ids:
            try:
                inner_ids.append(trainset.to_inner_iid(raw_id))
                raw_ids_known.append(raw_id)
            except ValueError:
                continue  # movie never seen in training -> skip for CF scoring

        if not inner_ids:
            return {}

        inner_ids = np.array(inner_ids)

        if user_known:
            scores = (
                global_mean
                + self.cf_model.bu[inner_uid]
                + self.cf_model.bi[inner_ids]
                + self.cf_model.qi[inner_ids] @ self.cf_model.pu[inner_uid]
            )
        else:
            # No personalization possible -> item-bias-only fallback.
            # (This is the exact "quietly non-personalized" behavior worth
            # flagging rather than hiding -- see is_warm() above, which is
            # what the router should check *before* calling this.)
            scores = global_mean + self.cf_model.bi[inner_ids]

        return dict(zip(raw_ids_known, scores))

    # ---------------------------------------------------------
    # Recommend
    # ---------------------------------------------------------

    def recommend(self, user_id, top_n=10, exclude_movies=None):
        if exclude_movies is None:
            exclude_movies = []

        candidates = self.movies[~self.movies["movie_idx"].isin(exclude_movies)]
        candidate_ids = candidates["movie_idx"].tolist()

        score_map = self._score_all_candidates(user_id, candidate_ids)
        if not score_map:
            return pd.DataFrame()

        scored = candidates[candidates["movie_idx"].isin(score_map.keys())].copy()
        scored["predicted_rating"] = scored["movie_idx"].map(score_map)

        recommendations = (
            scored.sort_values(by="predicted_rating", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        columns = ["movie_idx", "title", "genres", "overview", "vote_average", "predicted_rating"]
        return recommendations[columns]