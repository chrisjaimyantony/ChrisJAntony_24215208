# Practical 3: Implement a Feedforward Neural Network using PyTorch

## Aim

To implement a simple Feedforward Neural Network (FNN) using **PyTorch** and train it to learn the logical **AND** gate.

---

## Objective

- Understand the architecture of a Feedforward Neural Network.
- Learn how to build a neural network using PyTorch.
- Train the model using the AND gate dataset.
- Observe how the network learns through backpropagation.
- Compare the model's predictions after training.

---

## Theory

A **Feedforward Neural Network (FNN)** is one of the most fundamental types of Artificial Neural Networks. Information flows in only one direction—from the input layer through one or more hidden layers to the output layer. Unlike recurrent neural networks, there are no feedback connections or cycles.

The network learns by adjusting its weights using the **Backpropagation** algorithm, while an optimization algorithm such as **Adam** updates the weights to minimize the prediction error.

In this experiment, the neural network is trained to learn the **AND Gate**, whose truth table is shown below.

| Input A | Input B | Output |
|:-------:|:-------:|:------:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

---

## Libraries Used

| Library | Purpose |
|----------|---------|
| PyTorch (`torch`) | Build and train the neural network |
| `torch.nn` | Create neural network layers and activation functions |
| `torch.optim` | Optimize model parameters using optimization algorithms |

---

## Model Architecture

```text
Input Layer (2 Neurons)
        │
        ▼
Hidden Layer (4 Neurons, ReLU)
        │
        ▼
Output Layer (1 Neuron, Sigmoid)
```

### Layer Description

### Input Layer

- Accepts two input values.
- Represents the two inputs of the AND gate.

### Hidden Layer

- Contains **4 neurons**.
- Uses the **ReLU activation function**.
- Learns meaningful features from the input data.

### Output Layer

- Contains **1 neuron**.
- Uses the **Sigmoid activation function**.
- Produces an output probability between **0 and 1**.

---

## Algorithm

1. Import the required PyTorch libraries.
2. Create the AND gate dataset.
3. Define the Feedforward Neural Network.
4. Initialize the model.
5. Define the Binary Cross-Entropy loss function.
6. Initialize the Adam optimizer.
7. Perform forward propagation.
8. Compute the loss.
9. Perform backpropagation and update the weights.
10. Repeat the training process for 100 epochs.
11. Predict the outputs using the trained model.
12. Convert probabilities into binary class labels.

---

## Code Explanation

### Step 1 — Import Libraries

Import PyTorch along with the neural network and optimization modules required to build and train the model.

---

### Step 2 — Create Dataset

The AND gate dataset consists of four possible input combinations and their corresponding expected outputs.

---

### Step 3 — Define the Neural Network

A custom neural network class is created by inheriting from `nn.Module`.

The model contains:

- Input Layer (2 features)
- Hidden Layer (4 neurons with ReLU activation)
- Output Layer (1 neuron with Sigmoid activation)

---

### Step 4 — Initialize the Model

Create an instance of the neural network and initialize the loss function and optimizer.

The model uses:

- **Optimizer:** Adam
- **Loss Function:** Binary Cross Entropy Loss (BCELoss)

---

### Step 5 — Forward Propagation

The input data is passed through the network to produce output predictions.

The flow of computation is:

```text
Input
   │
Linear Layer
   │
ReLU
   │
Linear Layer
   │
Sigmoid
   │
Prediction
```

---

### Step 6 — Train the Model

The model is trained for **100 epochs**.

During each epoch:

- Perform forward propagation.
- Compute the loss.
- Clear previous gradients.
- Perform backpropagation.
- Update the network weights.

This process gradually improves the model's predictions.

---

### Step 7 — Predict the Outputs

After training, predictions are generated using the trained model.

The sigmoid output is converted into binary class labels using a threshold of **0.5**.

- Output ≥ 0.5 → Class 1
- Output < 0.5 → Class 0

---

## Expected Output

After training, the network successfully learns the AND gate.

Example:

```text
Predictions:

Input: [0.0, 0.0] → Probability: 0.01 → Class: 0
Input: [0.0, 1.0] → Probability: 0.03 → Class: 0
Input: [1.0, 0.0] → Probability: 0.02 → Class: 0
Input: [1.0, 1.0] → Probability: 0.98 → Class: 1
```

*Note:* The exact probability values may differ because the network weights are initialized randomly before training.

---

## Result

The Feedforward Neural Network successfully learns the logical **AND** operation using PyTorch.

After training, the model correctly classifies all four input combinations, demonstrating how a neural network learns patterns by adjusting its weights through backpropagation and optimization.

---

## Applications

- Binary Classification
- Pattern Recognition
- Medical Diagnosis
- Fraud Detection
- Spam Detection
- Credit Risk Prediction
- Decision Support Systems

---

## Conclusion

This experiment demonstrates the implementation of a basic Feedforward Neural Network using **PyTorch**. It covers the complete workflow of creating a neural network, defining the loss function, training the model using backpropagation and the Adam optimizer, and evaluating the trained network. Although the dataset is small, the experiment effectively illustrates the fundamental principles of supervised learning and neural network training.

---

## References

1. PyTorch Documentation — https://pytorch.org/docs/stable/
2. PyTorch Tutorials — https://pytorch.org/tutorials/
3. Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning*, MIT Press.
4. Simon J.D. Prince, *Understanding Deep Learning*, MIT Press.