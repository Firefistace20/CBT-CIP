# CIFAR-10 Image Classification with CNNs

## Introduction
This project demonstrates the use of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow and Keras. It provides functionality to interactively classify images and display their predicted and true labels. Additionally, the program includes filtering for specific classes like "horse," "dog," and "deer."

---

## Dataset
- *Name:* CIFAR-10
- *Description:* A widely-used dataset consisting of 60,000 32x32 color images across 10 classes.
- *Classes:*
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck
- *Preprocessing:*
  - Images are normalized to values between 0 and 1 by dividing pixel values by 255.
  - Labels are integers mapped to the class names.

---

## Features
1. *Training and Model Building:*
   - Builds a CNN model with three convolutional layers and two max-pooling layers.
   - Includes options to load a pre-trained model if it exists or train a new one from scratch.

2. *Interactive Image Classification:*
   - Allows users to input an index to classify and visualize any test image.
   - Displays the true label and the predicted label for the selected image.

3. *Class Filtering:*
   - Focused analysis on specific classes: "horse," "dog," and "deer."
   - Extracts and displays images corresponding to these classes.

---

## How to Run the Project

### Prerequisites
- Python 3.x
- Required Libraries:
  - TensorFlow
  - Matplotlib
  - Numpy (optional, if not included in TensorFlow dependencies)

### Steps to Run
1. *Clone the Repository:*
   bash
   git clone <repository-url>
   cd <repository-folder>
   

2. *Install Dependencies:*
   bash
   pip install tensorflow matplotlib
   

3. *Run the Script:*
   Execute the Python file in your preferred environment (e.g., terminal, Jupyter Notebook, or Google Colab).
   bash
   python cifar10_cnn.py
   

4. *Interactive Classification:*
   - Enter an index (0–9999) when prompted to classify an image from the test set.
   - Press Ctrl+C to exit the interactive loop.

---

## Code Explanation

### Key Functions
1. *classify_image(image)*
   - Expands the input image dimensions to match the model's batch input requirements.
   - Predicts the class of the image and returns the class name.

2. *show_image_with_prediction(image, true_label)*
   - Displays the input image with its predicted and true labels.

3. *show_single_image_with_prediction(image, true_label_index)*
   - Displays the image and converts the true label index into its class name for clarity.

4. *Interactive Loop*
   - Allows users to interactively classify images by entering indices.
   - Gracefully handles invalid inputs and exits on Ctrl+C.

---

## Model Architecture
- *Layer 1:* Conv2D (32 filters, 3x3 kernel, ReLU activation)
- *Layer 2:* MaxPooling2D (2x2 pool size)
- *Layer 3:* Conv2D (64 filters, 3x3 kernel, ReLU activation)
- *Layer 4:* MaxPooling2D (2x2 pool size)
- *Layer 5:* Conv2D (64 filters, 3x3 kernel, ReLU activation)
- *Flatten Layer*
- *Dense Layer 1:* 64 units, ReLU activation
- *Dense Layer 2:* 10 units (output), no activation

---

## Example Output
- *Predicted:* Horse
- *True:* Horse  
![Example Image](example_image.png)

---

## Acknowledgments
- CIFAR-10 dataset from [Kaggle](https://www.kaggle.com/c/cifar-10) and TensorFlow's keras.datasets.
- TensorFlow and Keras documentation for guidance on model building.

---

## Future Improvements
- Extend support to other datasets.
- Enhance the CNN architecture for higher accuracy.
- Add a graphical user interface (GUI) for easier image selection and classification.
