
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
    if os.path.exists(TRAIN_INTERACTION_COUNTS_PATH):
        with open(TRAIN_INTERACTION_COUNTS_PATH, "rb") as file:
            return pickle.load(file)

    if os.path.exists(TRAIN_INTERACTIONS_CSV_PATH):
        train_df = pd.read_csv(TRAIN_INTERACTIONS_CSV_PATH)
        return train_df["user_id"].value_counts().to_dict()

    st.warning(
        "Every user will be treated as cold."
    )
    return {}


@st.cache_resource
def load_embed_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.warning(f"Could not load embedding model ({e})."
                   " Genre search will use simple keyword matching instead."
                   "(We skipped a larger AI library here to keep the app light and fast on our current hosting —"
                   "keyword-based genre matching is used instead of deep semantic search."
                   )
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