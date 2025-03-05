import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=10000, ngram_range=(1,2), stop_words="english"):
        """
        Custom TF-IDF vectorizer with optimizations.
        - max_features: Limits vocabulary size to the most important words.
        - ngram_range: (1,1) for unigrams, (1,2) for bigrams, etc.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words)


    def fit(self, X, y=None):
        """Fit the vectorizer to the text data."""
        if isinstance(X, pd.DataFrame) and "text" in X.columns:
            X = X["text"]
        self.vectorizer.fit(X)
        return self


    def transform(self, X):
        """Transforms text data into TF-IDF feature vectors."""
        if isinstance(X, pd.DataFrame) and "text" in X.columns:
            X = X["text"]
        vec_result = self.vectorizer.transform(X)
        return vec_result.toarray()  # Ensures Output is Numeric
