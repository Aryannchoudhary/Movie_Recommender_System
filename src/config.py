import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# API Keys
OMDB_API_KEY = os.getenv("OMDB_API_KEY")


# Paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")

MOVIES_PATH = os.path.join(MODELS_DIR, "movies.pkl")
FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "movie_index.faiss")
INDICES_PATH = os.path.join(MODELS_DIR, "indices.pkl")
CF_MODEL_PATH = os.path.join(MODELS_DIR, "cf_model.pkl")
TRAIN_INTERACTION_COUNTS_PATH = os.path.join(MODELS_DIR, "train_interaction_counts.pkl")
TRAIN_INTERACTIONS_CSV_PATH = os.path.join(MODELS_DIR, "train_interactions.csv")


# Model Configuration

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_N_RECOMMENDATIONS = 10

MIN_INTERACTIONS_WARM = 5


# OMDb API
OMDB_BASE_URL = "https://www.omdbapi.com/"