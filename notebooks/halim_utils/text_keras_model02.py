from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


def build_keras_model(input_dim, l2_reg=0.01, dropout1=0.5, dropout2=0.5, dropout3=0.4, learning_rate=0.0001):
    model = Sequential([
        Input(shape=(input_dim,)),  # Input dim = tfidf max feature
        Dense(512, kernel_regularizer=l2(l2_reg)),
        LeakyReLU(negative_slope=0.02), Dropout(0.5),
        Dense(256),
        LeakyReLU(negative_slope=0.01),Dropout(0.5),
        Dense(128),
        LeakyReLU(negative_slope=0.01), Dropout(0.4),
        Dense(64),
        LeakyReLU(negative_slope=0.01),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision"),
                 ])
    return model


input_dim = 10000

es = EarlyStopping(monitor="loss", patience=30, restore_best_weights=True)

keras_wrapper = KerasClassifier(
    model=build_keras_model,
    input_dim=input_dim,
    l2_reg=0.01,
    dropout1=0.5,
    dropout2=0.5,
    dropout3=0.4,
    learning_rate=0.0001,
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1)
