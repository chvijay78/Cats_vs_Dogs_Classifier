# 🐱🐶 Cats vs Dogs Image Classifier

This project implements a deep learning-based **binary image classifier** that distinguishes between cats and dogs using **Convolutional Neural Networks (CNNs)** built with **TensorFlow/Keras**. It utilizes the `cats_vs_dogs` dataset from TensorFlow Datasets and includes a web interface for image prediction.

---

## 📌 Features

- ✅ Automatically loads and preprocesses data using `tensorflow_datasets`
- 🧠 CNN model built from scratch using `TensorFlow/Keras`
- 📊 Visualizes training and validation accuracy/loss during training
- 📷 Predicts whether an uploaded image is a **cat or a dog**
- 💡 Easily extendable for custom datasets or further enhancements

---

## 📦 Requirements

Install the necessary libraries:

```bash
pip install tensorflow matplotlib streamlit tensorflow-datasets numpy pillow
