"""
preprocess.py
-------------
NLP text preprocessing pipeline for Spam Email Classifier.
Handles: lowercasing, punctuation removal, stopword removal,
         tokenization, stemming, and missing value handling.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data (runs once)
def download_nltk_data():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_data()

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline for a single text string.

    Steps:
        1. Handle None / NaN values
        2. Convert to lowercase
        3. Remove URLs
        4. Remove punctuation & special characters
        5. Tokenize
        6. Remove stopwords
        7. Apply Porter Stemming

    Args:
        text (str): Raw email / SMS text.

    Returns:
        str: Cleaned and stemmed text string.
    """
    # Step 1 – Handle missing values
    if pd.isna(text) or text is None:
        return ""

    # Step 2 – Lowercase
    text = str(text).lower()

    # Step 3 – Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Step 4 – Remove punctuation and special characters (keep letters & digits)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Step 5 – Tokenize
    tokens = word_tokenize(text)

    # Step 6 – Remove stopwords and short tokens
    tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]

    # Step 7 – Stemming
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load the spam.csv dataset and apply full preprocessing pipeline.

    Dataset format:
        v1 → label  ('ham' or 'spam')
        v2 → text   (email / SMS content)

    Args:
        filepath (str): Path to spam.csv file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns:
                      ['label', 'text', 'clean_text', 'label_encoded']
    """
    # Load CSV
    df = pd.read_csv(filepath, encoding='latin-1')

    # Keep only relevant columns (handles extra unnamed cols from Kaggle export)
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'text']

    # Drop rows where text is missing
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Apply NLP cleaning
    print(f"[INFO] Preprocessing {len(df)} messages...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Encode labels: ham → 0, spam → 1
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

    print(f"[INFO] Preprocessing complete.")
    print(f"       Ham  (0): {(df['label_encoded'] == 0).sum()}")
    print(f"       Spam (1): {(df['label_encoded'] == 1).sum()}")

    return df


def get_word_frequencies(df: pd.DataFrame, label: str, top_n: int = 20) -> dict:
    """
    Get the most frequent words for a given class (ham or spam).

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        label (str): 'ham' or 'spam'.
        top_n (int): Number of top words to return.

    Returns:
        dict: {word: frequency} sorted descending.
    """
    subset = df[df['label'] == label]['clean_text']
    all_words = ' '.join(subset).split()
    freq = {}
    for word in all_words:
        freq[word] = freq.get(word, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
