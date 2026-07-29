import re

from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.config import TOP_N_RECOMMENDATIONS, MIN_INTERACTIONS_WARM


class HybridRecommender:
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

    # Popular Movies

    def popularity_recommend(self, top_n=TOP_N_RECOMMENDATIONS):
        columns = ["movie_idx", "title", "genres", "overview", "vote_average", "vote_count", "weighted_score"]

        if "weighted_score" in self.movies.columns:
            return (
                self.movies.sort_values(by="weighted_score", ascending=False)[columns]
                .head(top_n)
                .reset_index(drop=True)
            )

        return self.movies.sort_values(by="vote_average", ascending=False).head(top_n).reset_index(drop=True)

    def genre_recommend(self, genres, top_n=TOP_N_RECOMMENDATIONS):
        if isinstance(genres, str):
            genres = [genres]
        pattern = "|".join(re.escape(g) for g in genres if g)
        if not pattern:
            return self.popularity_recommend(top_n)

        filtered = self.movies[self.movies["genres"].str.contains(pattern, case=False, na=False, regex=True)]

        if filtered.empty:
            return self.popularity_recommend(top_n)

        sort_column = "weighted_score" if "weighted_score" in filtered.columns else "vote_average"
        return filtered.sort_values(by=sort_column, ascending=False).head(top_n).reset_index(drop=True)

    # Automatic router

    def recommend(
        self,
        user_id=None,
        movie_title=None,
        genre_profile=None,
        exclude_movies=None,
        top_n=TOP_N_RECOMMENDATIONS,
        force_strategy=None,
    ):
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
        if genre_profile:
            # Prefer the semantic embedding-based genre match if an embed
            semantic_results = self.content.recommend_from_genres(genre_profile, top_n)
            if not semantic_results.empty:
                return {"strategy": "Content-Based (genre profile, cold-start)", "results": semantic_results}
            return {
                "strategy": "Genre-Based (cold-start fallback)",
                "results": self.genre_recommend(genre_profile, top_n),
            }

        return {"strategy": "Popularity-Based (no signal available)", "results": self.popularity_recommend(top_n)}