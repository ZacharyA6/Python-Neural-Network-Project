# Python-Neural-Network-Project
Star Wars Character Classifier
Overview

This project is a small neural network that classifies a Star Wars character from a short text description. It trains on a custom dataset of character descriptions, saves the trained model, and lets the user type their own prompt to get the top predicted matches.

Files
dataset.py - stores the training text, labels, label mappings, and loads extra character info from characters.json
model.py - builds, trains, and saves the TensorFlow/Keras text classification model
run.py - loads the saved model and runs the interactive character classifier
characters.json - stores faction, affiliations, and a fun fact for each character

How It Works
Training descriptions and labels are loaded from dataset.py.
TextVectorization converts text into integer sequences.
The model uses an Embedding layer, GlobalAveragePooling1D, and dense layers to classify the text.
The trained model is saved as starwars_text_model.keras.
run.py loads the model, predicts the best match, and prints extra character details.

Characters Included:
Luke Skywalker
Darth Vader
Leia Organa
Boba Fett
Obi-Wan Kenobi
Padme Amidala
Ahsoka Tano
Emperor Palpatine
Han Solo
Darth Maul

How to Run:
Train the Model: run model.py
Run the Classifier: run.py

Notes:
The model is trained only on the custom descriptions in dataset.py, so predictions depend heavily on how similar the input is to the training text.
characters.json adds extra output details after prediction, such as faction and fun facts.
