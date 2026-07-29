# import pickle
# import faiss
# import streamlit as st

# from src.config import (
#     MOVIES_PATH,
#     FAISS_INDEX_PATH,
#     INDICES_PATH,
#     CF_MODEL_PATH,
# )


# @st.cache_resource
# def load_movies():
#     with open(MOVIES_PATH, "rb") as file:
#         return pickle.load(file)


# @st.cache_resource
# def load_faiss_index():
#     return faiss.read_index(FAISS_INDEX_PATH)


# @st.cache_resource
# def load_indices():
#     with open(INDICES_PATH, "rb") as file:
#         return pickle.load(file)


# @st.cache_resource
# def load_cf_model():
#     with open(CF_MODEL_PATH, "rb") as file:
#         return pickle.load(file)


# @st.cache_resource
# def load_all_models():
#     return {
#         "movies": load_movies(),
#         "faiss_index": load_faiss_index(),
#         "indices": load_indices(),
#         "cf_model": load_cf_model(),
#     }





import os
import pickle

import faiss
import pandas as pd
import streamlit as st

from src.config import (
    MOVIES_PATH,
    FAISS_INDEX_PATH,
    INDICES_PATH,
    CF_MODEL_PATH,
    TRAIN_INTERACTION_COUNTS_PATH,
    TRAIN_INTERACTIONS_CSV_PATH,
    EMBEDDING_MODEL,
)


@st.cache_resource
def load_movies():
    with open(MOVIES_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_PATH)


@st.cache_resource
def load_indices():
    with open(INDICES_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_cf_model():
    with open(CF_MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_train_interaction_counts():
    """
    Needed for the hybrid router's warm/cold check. Tries a few things,
    in order, so this doesn't hard-crash the whole app if it's missing:

    1. A precomputed {user_id: count} dict pickle (fastest, if you saved one)
    2. The raw train interactions CSV -> derive counts with value_counts()
    3. Neither found -> return {} (every user will be treated as cold,
       which is a safe default, just less useful than having real counts)
    """
    if os.path.exists(TRAIN_INTERACTION_COUNTS_PATH):
        with open(TRAIN_INTERACTION_COUNTS_PATH, "rb") as file:
            return pickle.load(file)

    if os.path.exists(TRAIN_INTERACTIONS_CSV_PATH):
        train_df = pd.read_csv(TRAIN_INTERACTIONS_CSV_PATH)
        return train_df["user_id"].value_counts().to_dict()

    st.warning(
        "No train interaction counts found (checked both "
        f"{os.path.basename(TRAIN_INTERACTION_COUNTS_PATH)} and "
        f"{os.path.basename(TRAIN_INTERACTIONS_CSV_PATH)}). "
        "Every user will be treated as cold until this is added."
    )
    return {}


@st.cache_resource
def load_embed_model():
    """
    Needed for the genre-profile cold-start path (embedding a synthetic
    'taste query' from a user's stated genre preferences). If this fails to
    load (e.g. no internet on first run to fetch the model), the app still
    works -- it just falls back to simple substring genre filtering instead
    of semantic genre matching.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.warning(f"Could not load embedding model ({e}). Genre search will use simple keyword matching instead.")
        return None


@st.cache_resource
def load_all_models():
    return {
        "movies": load_movies(),
        "faiss_index": load_faiss_index(),
        "indices": load_indices(),
        "cf_model": load_cf_model(),
        "train_interaction_counts": load_train_interaction_counts(),
        "embed_model": load_embed_model(),
    }