import pickle
import faiss
import streamlit as st

from src.config import (
    MOVIES_PATH,
    FAISS_INDEX_PATH,
    INDICES_PATH,
    CF_MODEL_PATH,
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
def load_all_models():
    return {
        "movies": load_movies(),
        "faiss_index": load_faiss_index(),
        "indices": load_indices(),
        "cf_model": load_cf_model(),
    }