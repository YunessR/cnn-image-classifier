# Convolutional Neural Network in PyTorch

A small convolutional neural network implemented in PyTorch. The project performs several convolution and ReLU stages, flattens the result, applies two linear layers, and uses softmax to produce a digit prediction and confidence.

## Project Structure

```text
cnn-project/
├── cnn.py
├── weight0.pt
├── weight1.pt
├── weight2.pt
├── weight3.pt
├── weight4.pt
├── bias0.pt
├── bias1.pt
├── bias2.pt
├── bias3.pt
├── bias4.pt
├── examples/
│   └── example_test_digit_*.png
├── validation/
│   └── validate_test_digit_*.png
└── README.md
```

## Requirements

- Python 3
- PyTorch
- NumPy
- Pillow

Install the Python dependencies with:

```bash
pip install torch numpy pillow
```

## Running the Project

From the project root:

```bash
python cnn.py examples/example_test_digit_0.png
```

The program prints the predicted digit and its confidence.

## Model Files

The `.pt` files contain the weights and biases loaded by `cnn.py`. They are kept in the project root so the program can load them using the filenames already used by the source code.

## Notes

The source code in `cnn.py` is preserved as-is from the original implementation. The project organization and filenames were changed only to make the repository easier to read and present as a standalone project.
