# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()


# # API Keys
# OMDB_API_KEY = os.getenv("OMDB_API_KEY")


# # Paths

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODELS_DIR = os.path.join(BASE_DIR, "models")

# MOVIES_PATH = os.path.join(MODELS_DIR, "movies.pkl")
# FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "movie_index.faiss")
# INDICES_PATH = os.path.join(MODELS_DIR, "indices.pkl")
# CF_MODEL_PATH = os.path.join(MODELS_DIR, "cf_model.pkl")
# TRAIN_INTERACTIONS_PATH = os.path.join(
#     MODELS_DIR,
#     "train_interaction_counts.pkl"
# )


# # Model Configuration

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# TOP_N_RECOMMENDATIONS = 10


# # OMDb API
# OMDB_BASE_URL = "https://www.omdbapi.com/"






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

# Either of these may exist -- loader.py checks both so you don't need to
# regenerate artifacts if you already have one or the other:
#   - a precomputed {user_id: interaction_count} dict pickle, OR
#   - the raw train interactions table (CSV), from which counts are derived
TRAIN_INTERACTION_COUNTS_PATH = os.path.join(MODELS_DIR, "train_interaction_counts.pkl")
TRAIN_INTERACTIONS_CSV_PATH = os.path.join(MODELS_DIR, "train_interactions.csv")


# Model Configuration

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_N_RECOMMENDATIONS = 10

# A user needs at least this many training interactions before Collaborative
# Filtering is trusted to be personalized for them; below this, they're
# treated as "cold" and routed to the content-based fallback instead.
MIN_INTERACTIONS_WARM = 5


# OMDb API
OMDB_BASE_URL = "https://www.omdbapi.com/"