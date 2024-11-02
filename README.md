# CNN-Handwritten-Digit-Classification
This project aims to classify handwritten digits using the MNIST dataset and Convolutional Neural Networks (CNNs). It serves as an excellent introduction to deep learning and image classification tasks.

# Introduction
In this project, we develop a model to identify handwritten digits from images. The MNIST dataset, which contains images of handwritten digits (0-9), is used for training and testing the model. This project demonstrates the power of CNNs in image recognition tasks.

# Dataset
The dataset used in this project is the MNIST dataset, comprising 70,000 grayscale images of handwritten digits. Each image is 28x28 pixels in size.

# Installation
To run this project, ensure you have Python installed along with the following libraries:

  1) TensorFlow
  2) Keras
  3) NumPy
  4) Matplotlib
  5) scikit-learn

  # You can install the necessary packages using pip: pip install tensorflow keras numpy matplotlib scikit-learn

# Model Architecture
The model consists of multiple convolutional layers followed by max-pooling layers, a flattening layer, and dense layers. The final layer uses softmax activation to classify the digits.
