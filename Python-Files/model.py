import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from dataset import X, y, label_to_id

def build_and_train_model():
    # Number of classes = number of unique labels used in training
    num_classes = len(label_to_id)
    max_tokens = 2000
    max_len = 40

    # Convert texts to TF string tensor
    X_tensor = tf.constant(X)

    # ---- Text vectorizer ----
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_len,
    )

    # Learn vocabulary from training texts
    text_ds = tf.data.Dataset.from_tensor_slices(X_tensor).batch(32)
    vectorizer.adapt(text_ds)

    # ---- Simple classifier model ----
    model = keras.Sequential([
        vectorizer,                                  # text -> int sequence
        keras.layers.Embedding(max_tokens, 64),      # learn token embeddings
        keras.layers.GlobalAveragePooling1D(),       # average over time
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ---- Train ----
    model.fit(
        X_tensor,
        y,
        epochs=30,
        batch_size=4,
        verbose=1,
    )

    # ---- Save model ----
    model.save("starwars_text_model.keras")
    print("Model saved as starwars_text_model.keras")

if __name__ == "__main__":
    build_and_train_model()