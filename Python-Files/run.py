import tensorflow as tf
from tensorflow import keras
from dataset import id_to_label, CHAR_INFO

# Load trained model (includes TextVectorization inside)
model = keras.models.load_model("starwars_text_model.keras")

def predict_character(text, top_k=3):
    # Single-string TF tensor input
    x = tf.constant([text])
    probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)

    # Get indices of top-k probabilities
    top_ids = probs.argsort()[-top_k:][::-1]
    return [(id_to_label[i], float(probs[i])) for i in top_ids]

def main():
    print("\n Star Wars Character Classifier")
    print("Type a description of a Star Wars character and I'll guess who it is!")
    print("Example: 'Sith Lord in black armor with a red lightsaber'\n")

    while True:
        user_input = input("Describe a character (or type 'quit' to exit): ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        preds = predict_character(user_input, top_k=3)
        best_name, best_prob = preds[0]

        # Look up facts – CHAR_INFO keys must match labels (e.g. "Luke Skywalker", "Boba Fett", etc.)
        info = CHAR_INFO.get(best_name, {})

        print(f"\n I think you're describing **{best_name}** ({best_prob * 100:.1f}% confidence)")

        faction = info.get("faction", "Unknown")
        affiliations = info.get("affiliations", [])
        fun_fact = info.get("fun_fact", "No fun fact available.")

        print(f"   • Faction: {faction}")
        print(f"   • Affiliations: {', '.join(affiliations) if affiliations else 'None'}")
        print(f"   • Fun fact: {fun_fact}")

        print("\n Other possible matches:")
        for name, p in preds[1:]:
            print(f"   • {name} ({p * 100:.1f}%)")

        print("\n-------------------------------\n")

if __name__ == "__main__":
    main()