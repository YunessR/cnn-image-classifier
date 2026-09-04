Convolutional Neural Network for MNIST Classification
A PyTorch implementation of a simple CNN trained to classify handwritten digits from the MNIST dataset. This project demonstrates core deep learning concepts: convolutional layers, activation functions, and fully-connected classification heads.

Model Architecture
Input: 28×28 grayscale images
Convolution Blocks: Multiple conv + ReLU stages with pooling
Classification Head: Flatten → Dense layers → Softmax output
Output: 10-class digit prediction (0-9)
