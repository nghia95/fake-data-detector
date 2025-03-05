from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from scikeras.wrappers import KerasClassifier


def build_keras_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),  # Input dim = tfidf max feature
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification (AI vs. Human)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_dim = 10000

# Wrap the model in KerasClassifier (for compatibility with scikit-learn pipeline)
keras_wrapper = KerasClassifier(
    model=build_keras_model,
    input_dim=input_dim,
    epochs=10, batch_size=32, verbose=1)
