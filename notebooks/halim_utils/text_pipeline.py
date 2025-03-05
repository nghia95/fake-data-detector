import pandas as pd
import string
import re
import numpy as np
from sklearn.pipeline import Pipeline
from text_cleaner import TextCleaner
from text_tfidf import TFIDFVectorizer
from text_keras_model import keras_wrapper
from sklearn.exceptions import NotFittedError


final_pipeline = Pipeline([
    ("text_cleaning", TextCleaner()),
    ("tfidf", TFIDFVectorizer(max_features=10000, stop_words="english")),
    ("classifier", keras_wrapper)
])


# Predict Text (UI)
class TextClassificationModel:
    def __init__(self, pipeline):
        """Wrap the pre-trained pipeline."""
        self.pipeline = pipeline

    def predict(self, input_text: str):
        """Predict whether the input text is AI or Human."""
        # Convert input into DataFrame (expected by pipeline)
        input_df = pd.DataFrame({"text": [input_text]})

        try:
            # Apply transformation separately if needed
            input_df = self.pipeline[:-1].transform(input_df)  # Process text excluding classifier

            # Make prediction using the classifier
            y_pred_proba = self.pipeline.steps[-1][1].predict_proba(input_df)

            # Ensure y_pred_proba is a 2D array (even for a single sample)
            if y_pred_proba.ndim == 1:
                y_pred_proba = y_pred_proba.reshape(1, -1)

            # Extract probability for AI class (assumed to be at index 1)
            ai_probability = y_pred_proba[0, 1]  # First row, second column

            # Determine prediction and format probability
            prediction = "AI" if ai_probability > 0.5 else "Human"
            probability = f"{ai_probability * 100:.2f}%"

            return {"Prediction": prediction, "AI Probability": probability}

        except Exception as e:
            # Handle potential errors (e.g., pipeline not fitted, invalid input)
            return {"Error": str(e)}

# Create model instance
model = TextClassificationModel(final_pipeline)
