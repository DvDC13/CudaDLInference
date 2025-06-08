# CudaDLInference (CUDA CNN Inference Engine (LeNet))

This project implements a custom Convolutional Neural Network (CNN) inference engine in CUDA, specifically designed to run the LeNet architecture efficiently on the GPU.

## 🚀 Project Goals

- Execute full inference of a pretrained LeNet model using CUDA.
- Support key CNN features:
  - 2D convolutions with padding, stride
  - ReLU activation
  - Max pooling
  - Flatten
  - Fully connected (dense) layers
- Compare performance and accuracy with PyTorch

## 🧠 Model Used

The model used is **LeNet**, exported from PyTorch and converted to raw binary format `.bin` for fast loading inside the custom engine.

Example input: 1x1x28x28 grayscale images.  
Output: 10-class classification logits.

## 🛠️ How to Build

You need:

- CUDA toolkit
- CMake
- Python
