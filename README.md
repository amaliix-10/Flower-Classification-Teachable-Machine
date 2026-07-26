# Flower-Classification-Using-Teachable-Machine
# 🌻 Sunflower vs. Lily 🌸 Image Classification

An end-to-end Machine Learning project using **Teachable Machine by Google** and **TensorFlow / Keras** to classify images into two flower categories: **Sunflower** and **Lily**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Dataset & Classes](#-dataset--classes)
- [Prerequisites & Requirements](#-prerequisites--requirements)
- [File Structure](#-file-structure)
- [How to Run (Google Colab / Local Environment)](#-how-to-run)
- [Code Explanation](#-code-explanation)
- [Output Example](#-output-example)

---

## 🚀 Overview
This project demonstrates the workflow of training a custom Computer Vision classification model using Teachable Machine, exporting the trained weights in Keras format, and executing inference on unseen test images using Python.

---

## 🏗️ Project Architecture
1. **Model Training:** Teachable Machine (Standard Image Model).
2. **Export Format:** Keras (`keras_model.h5` + `labels.txt`).
3. **Execution Environment:** Python 3 (Google Colab / Jupyter Notebook).
4. **Input Data:** Resized and normalized image (224 x 224 x 3).

---

## 🏷️ Dataset & Classes
The model is trained on **two target classes**:
1. `0 Sunflower` 🌻
2. `1 Lily` 🌸

---

## 🛠️ Prerequisites & Requirements
Make sure you have the required Python packages installed:

```bash
pip install tensorflow tf-keras pillow numpy
