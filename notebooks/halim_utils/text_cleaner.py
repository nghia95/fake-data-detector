import pandas as pd
import string
import re
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    def text_cleaning(self, text):
        if not isinstance(text, str):
            text = str(text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra spaces
        text = ' '.join(text.split())
        return text.strip()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Apply cleaning to text column."""
        if isinstance(X, pd.DataFrame):
            if "text" in X.columns:
                return X.assign(text=X["text"].apply(self.text_cleaning))
            else:
                raise ValueError("Expected DataFrame with 'text' column.")
        elif isinstance(X, str):
            return self.text_cleaning(X)  # Handle single string input
        elif isinstance(X, list):
            return [self.text_cleaning(text) for text in X]  # Handle list of strings
        else:
            raise TypeError("Unsupported input type. Expected DataFrame, string, or list.")
