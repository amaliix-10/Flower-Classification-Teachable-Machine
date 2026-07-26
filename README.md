# Flower-Classification-Using-Teachable-Machine

An end-to-end Machine Learning project using **Teachable Machine by Google** and **TensorFlow / Keras** to classify images into two flower categories: **Sunflower** and **Lily**.
This project demonstrates the workflow of training a custom Computer Vision classification model using Teachable Machine, exporting the trained weights in Keras format, and executing inference on unseen test images using Python.

---
##  What is this project?

Have you ever wondered how your phone camera recognizes objects in real-time? 

This project is a mini **Artificial Intelligence (AI)** model that serves as an entry point into the world of **Computer Vision**. It takes an input image of a flower, analyzes its visual features and color patterns, and outputs:
1. **The predicted flower type** (Sunflower  or Lily ).
2. **The confidence score** (how certain the AI is about its prediction).

---

## How Was It Built?

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
 
##  File Structure

```text
├── keras_model.h5    # Trained Keras model weights and architecture
├── labels.txt        # Class labels (0 Sun Flower, 1 Lily Flower)
├── test.jpg          # Test image used for inference
├── predict.py        # Python script for running predictions
└── README.md         # Project documentation
```
---

##  How to Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `keras_model.h5`, `labels.txt`, and a test image (e.g., `test.jpg`) to the **Files** sidebar 📁.
3. Run the following Python script in a notebook cell:

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import tf_keras as tk

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tk.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("test.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```

----
##  Result & Execution Output

<img width="628" height="95" alt="Screenshot 2026-07-26 170156" src="https://github.com/user-attachments/assets/23016517-270a-45f0-91f7-21d0ec3f9a95" />

The above shows the actual execution of the inference script running on **Google Colab**. 

###  Console Output
```text
Class: Lily Flower
Confidence Score: 0.99997616
