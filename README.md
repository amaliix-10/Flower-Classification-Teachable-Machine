# Flower-Classification-Using-Teachable-Machine

An end-to-end Machine Learning project using **Teachable Machine by Google** and **TensorFlow / Keras** to classify images into two flower categories: **Sunflower** and **Lily**.
This project demonstrates the workflow of training a custom Computer Vision classification model using Teachable Machine, exporting the trained weights in Keras format, and executing inference on unseen test images using Python.

---
## 💡 What is this project?

Have you ever wondered how your phone camera recognizes objects in real-time? 

This project is a mini **Artificial Intelligence (AI)** model that serves as an entry point into the world of **Computer Vision**. It takes an input image of a flower, analyzes its visual features and color patterns, and outputs:
1. **The predicted flower type** (Sunflower  or Lily ).
2. **The confidence score** (how certain the AI is about its prediction).

---

## 🛠️ How Was It Built?

The project follows a complete 3-step Machine Learning workflow:

1. **Training:** We used **Google's Teachable Machine** platform, providing it with sample images for both flowers so the AI could learn the visual differences between them.
2. **Exporting:** We exported the trained AI weights and class labels in **TensorFlow / Keras** format (`keras_model.h5` and `labels.txt`).
3. **Prediction (Inference):** We wrote a Python script executed on **Google Colab** that loads the model, processes a new test image, and prints the final classification result.

---

##  Dataset & Classes
The model is trained on **two target classes**:
1. `0 Sunflower` 🌻
2. `1 Lily` 🌸

---
### Model Training & Evaluation (Teachable Machine Interface)

This screenshot demonstrates the training and validation phase using **Teachable Machine by Google**:

<img width="1816" height="863" alt="Screenshot 2026-07-26 135532" src="https://github.com/user-attachments/assets/dcd9e6a7-e658-4940-a9f7-6be915760a20" />


- **Classes Defined:**
  - `Sun Flower`: Class containing sample images of sunflowers.
  - `Lily Flower`: Class containing sample images of lily flowers.
- **Model Status:** Successfully trained (`Model Trained`).
- **Live Preview & Evaluation:**
  - An unseen test image of a pink **Lily** flower was fed into the preview panel.
  - The model accurately recognized the input image and predicted **Lily Flower with 100% confidence**, proving the correctness and efficiency of the trained classification model.
