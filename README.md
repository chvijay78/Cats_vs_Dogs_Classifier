# 🐱🐶 Cats vs Dogs Image Classifier

This project implements a deep learning-based binary image classifier that distinguishes between cats and dogs. Using Convolutional Neural Networks (CNNs) built with TensorFlow/Keras, the model is trained to achieve high accuracy on image data from the popular **Dogs vs. Cats** dataset.

## 📌 Features

- ✅ Uses `tensorflow_datasets` to load and preprocess data automatically
- 🧠 CNN model built from scratch using TensorFlow/Keras
- 📊 Visualizes training and validation accuracy/loss
- 📷 Predicts whether an image is a cat or a dog
- 💡 Easily extendable for improvements or deployment

---

## 📦 Requirements

Install the required libraries:
pip install tensorflow matplotlib (command)

📥 Dataset
We use the cats_vs_dogs dataset from TensorFlow Datasets:

import tensorflow_datasets as tfds
(train_data, val_data), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

Images are automatically labeled and pre-split for training and validation.
Preprocessing includes resizing and normalization.

🧠 Model Architecture
A simple CNN built using tf.keras.Sequential:

Conv2D → ReLU → MaxPooling
Conv2D → ReLU → MaxPooling
Flatten → Dense → Sigmoid

Easily customizable and suitable for binary classification tasks.

🏃 How to Run
1. Download the .h5 and app.py in a similar folder.
2. run these cmds in terminal or bash --
3. pip install streamlit tensorflow tensorflow-datasets numpy pillow
4. streamlit run app.py
5. Upload a cat or dog image
6. The app will predict the output as cat or dog with prediction score.



