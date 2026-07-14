# Practical 1: Perceptron Simulation using NumPy

## Aim
To implement a single-layer Perceptron using NumPy and train it to perform the logical **AND** operation.

## Description
A Perceptron is one of the simplest artificial neural networks. It consists of input weights, a bias, and an activation function. During training, the weights and bias are updated using the perceptron learning rule until the model correctly classifies the training data.

In this program:
- NumPy is used for mathematical operations.
- A **step activation function** produces binary outputs (0 or 1).
- The Perceptron is trained using the **AND gate** dataset.
- After training, the model predicts the output for all input combinations.

## Algorithm
1. Import the NumPy library.
2. Define the step activation function.
3. Create a `Perceptron` class with weights, bias, and learning rate.
4. Compute predictions using the weighted sum and activation function.
5. Update weights and bias during training based on prediction error.
6. Train the model using the AND gate dataset.
7. Display predictions for each input.

## Output
```
Predictions:
Input: [0 0], Output: 0
Input: [0 1], Output: 0
Input: [1 0], Output: 0
Input: [1 1], Output: 1
```

## Result
The Perceptron was successfully implemented and trained using NumPy. The trained model correctly learned the AND logic gate by producing the expected outputs for all input combinations.