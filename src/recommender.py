from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.config import TOP_N_RECOMMENDATIONS


class HybridRecommender:
    """
    Hybrid Movie Recommendation System

    Supported Strategies:
    ---------------------
    - content
    - collaborative
    - genre
    - popularity
    """

    def __init__(
        self,
        movies_df,
        faiss_index,
        indices,
        cf_model,
    ):

        self.movies = movies_df

        self.content = ContentBasedRecommender(
            movies_df=movies_df,
            faiss_index=faiss_index,
            indices=indices,
        )

        self.collaborative = CollaborativeRecommender(
            movies_df=movies_df,
            cf_model=cf_model,
        )

    # --------------------------------------------------------
    # Popular Movies
    # --------------------------------------------------------

    def popularity_recommend(self, top_n=TOP_N_RECOMMENDATIONS):

        columns = [
            "movie_idx",
            "title",
            "genres",
            "overview",
            "vote_average",
            "vote_count",
            "weighted_score",
        ]

        if "weighted_score" in self.movies.columns:

            return (
                self.movies
                .sort_values(
                    by="weighted_score",
                    ascending=False,
                )[columns]
                .head(top_n)
                .reset_index(drop=True)
            )

        return (
            self.movies
            .sort_values(
                by="vote_average",
                ascending=False,
            )
            .head(top_n)
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Genre Recommendation
    # --------------------------------------------------------

    def genre_recommend(
        self,
        genres,
        top_n=TOP_N_RECOMMENDATIONS,
    ):

        if isinstance(genres, str):
            genres = [genres]

        filtered = self.movies[
            self.movies["genres"].str.contains(
                "|".join(genres),
                case=False,
                na=False,
            )
        ]

        if filtered.empty:
            return self.popularity_recommend(top_n)

        sort_column = (
            "weighted_score"
            if "weighted_score" in filtered.columns
            else "vote_average"
        )

        return (
            filtered
            .sort_values(
                by=sort_column,
                ascending=False,
            )
            .head(top_n)
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Main Recommendation Router
    # --------------------------------------------------------

    def recommend(
        self,
        strategy="content",
        user_id=None,
        movie_title=None,
        genre=None,
        exclude_movies=None,
        top_n=TOP_N_RECOMMENDATIONS,
    ):

        if exclude_movies is None:
            exclude_movies = []

        strategy = strategy.lower()

        # ----------------------------------------
        # Collaborative Filtering
        # ----------------------------------------

        if strategy == "collaborative":

            return {
                "strategy": "Collaborative Filtering",
                "results": self.collaborative.recommend(
                    user_id=user_id,
                    top_n=top_n,
                    exclude_movies=exclude_movies,
                ),
            }

        # ----------------------------------------
        # Content-Based
        # ----------------------------------------

        if strategy == "content":

            return {
                "strategy": "Content-Based",
                "results": self.content.recommend(
                    movie_title=movie_title,
                    top_n=top_n,
                ),
            }

        # ----------------------------------------
        # Genre-Based
        # ----------------------------------------

        if strategy == "genre":

            return {
                "strategy": "Genre-Based",
                "results": self.genre_recommend(
                    genres=genre,
                    top_n=top_n,
                ),
            }

        # ----------------------------------------
        # Popularity-Based
        # ----------------------------------------

        return {
            "strategy": "Popularity-Based",
            "results": self.popularity_recommend(
                top_n=top_n,
            ),
        }