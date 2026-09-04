# Author: Yuness Rachidi
# Netid: Yir8
# Aid: The book and lectures were great resources. 
#      This video also was very helpful in understanding  and walking through how pyTorch's
#      Conv2d works: https://www.youtube.com/watch?v=j19Wdlu7Rtg&t=525s
import torch
import sys
import numpy
from PIL import Image


def conv2d(input_tensor, weights, bias, stride=2):

    in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weights.shape

    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    output = torch.zeros(out_channels, output_height, output_width)

    for out_channel in range(out_channels):
        for row in range(output_height):
            for col in range(output_width):

                sum_val = 0.0

                for in_channel in range(in_channels):
                    for kernel_row in range(kernel_height):
                        for kernel_col in range(kernel_width):

                            sum_val += (
                                input_tensor[in_channel,
                                             row * stride + kernel_row,
                                             col * stride + kernel_col]
                                *
                                weights[out_channel,
                                        in_channel,
                                        kernel_row,
                                        kernel_col]
                            )

                output[out_channel, row, col] = sum_val + bias[out_channel]

    return output


def main():

    # load image
    X = torch.tensor(
        numpy.array(Image.open(sys.argv[1])) / 255
    ).view(1, 28, 28).float()

    # load weights
    weight0 = torch.load("weight0.pt")
    weight1 = torch.load("weight1.pt")
    weight2 = torch.load("weight2.pt")
    weight3 = torch.load("weight3.pt")
    weight4 = torch.load("weight4.pt")

    # load biases
    bias0 = torch.load("bias0.pt")
    bias1 = torch.load("bias1.pt")
    bias2 = torch.load("bias2.pt")
    bias3 = torch.load("bias3.pt")
    bias4 = torch.load("bias4.pt")

    # first convolution
    z0 = conv2d(X, weight0, bias0)
    a0 = torch.relu(z0)

    # second convolution
    z1 = conv2d(a0, weight1, bias1)
    a1 = torch.relu(z1)

    # third convolution
    z2 = conv2d(a1, weight2, bias2)
    a2 = torch.relu(z2)

    # flatten
    flattened = a2.view(-1)

    # first linear layer
    z3 = flattened @ weight3.T + bias3

    # second linear layer
    z4 = z3 @ weight4.T + bias4

    # softmax decision
    decision = torch.nn.Softmax(dim=0)
    y_hat = decision(z4)

    prediction = torch.argmax(y_hat)
    confidence = y_hat[prediction]

    print(f"Prediction is {prediction} with confidence {confidence:3f}")


if __name__ == "__main__":
    main()