import requests
import streamlit as st

from src.config import OMDB_API_KEY, OMDB_BASE_URL


# ---------------------------------------------------------
# Cached API Request
# ---------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie(api_key: str, base_url: str, title: str):
    """
    Cached OMDb API request.
    Cache expires after 24 hours.
    """

    if not api_key:
        return {
            "success": False,
            "error": "OMDb API key not configured.",
        }

    try:

        response = requests.get(
            base_url,
            params={
                "apikey": api_key,
                "t": title,
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        if data.get("Response") == "False":

            return {
                "success": False,
                "error": data.get("Error", "Movie not found."),
                "poster": None,
            }

        return {
            "success": True,
            "title": data.get("Title", ""),
            "year": data.get("Year", ""),
            "genre": data.get("Genre", ""),
            "runtime": data.get("Runtime", ""),
            "director": data.get("Director", ""),
            "actors": data.get("Actors", ""),
            "plot": data.get("Plot", ""),
            "language": data.get("Language", ""),
            "country": data.get("Country", ""),
            "imdb_rating": data.get("imdbRating", "N/A"),
            "poster": data.get("Poster")
            if data.get("Poster") != "N/A"
            else None,
        }

    except requests.exceptions.RequestException as e:

        return {
            "success": False,
            "error": str(e),
            "poster": None,
        }


# ---------------------------------------------------------
# OMDb Client
# ---------------------------------------------------------

class OMDbClient:
    """
    Client for fetching movie metadata from OMDb.
    """

    def __init__(self):

        self.api_key = OMDB_API_KEY
        self.base_url = OMDB_BASE_URL

    def get_movie(self, title: str):
        """
        Fetch metadata for a movie title.
        """

        if not title:
            return {
                "success": False,
                "error": "Movie title is empty.",
                "poster": None,
            }

        return fetch_movie(
            self.api_key,
            self.base_url,
            title.strip(),
        )