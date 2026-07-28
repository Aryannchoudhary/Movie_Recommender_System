import re
import pandas as pd


def normalize_title(title: str) -> str:
    """
    Normalize movie title for matching.
    """
    if not title:
        return ""

    title = title.lower().strip()

    title = re.sub(r"\s+", " ", title)

    return title


def validate_user_id(user_id):
    """
    Validate user ID.
    """
    try:
        return int(user_id)
    except (TypeError, ValueError):
        return None


def format_rating(rating):
    """
    Format rating to one decimal place.
    """
    try:
        return f"{float(rating):.1f}"
    except (TypeError, ValueError):
        return "N/A"


def safe_value(value, default="N/A"):
    """
    Return default value if data is missing.
    """
    if value is None:
        return default

    if isinstance(value, float) and pd.isna(value):
        return default

    if str(value).strip() == "":
        return default

    return value


def dataframe_is_empty(df):
    """
    Check whether recommendation DataFrame is empty.
    """
    return df is None or df.empty