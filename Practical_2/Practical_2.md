# Practical 2: Implement a Feedforward Neural Network using TensorFlow

## Aim

To implement a simple Feedforward Neural Network (FNN) using TensorFlow and train it to learn the logical **AND** gate.

---

## Objective

- Understand the architecture of a Feedforward Neural Network.
- Learn how to build a neural network using TensorFlow and Keras.
- Train the model using the AND gate dataset.
- Compare the model's predictions before and after training.

---

## Theory

A **Feedforward Neural Network (FNN)** is one of the simplest forms of an Artificial Neural Network. Information moves in only one direction—from the input layer, through one or more hidden layers, to the output layer. There are no cycles or feedback connections.

The network learns by adjusting its weights during training using the **Backpropagation** algorithm along with an optimization algorithm such as **Adam**.

For this experiment, the network is trained to learn the **AND Gate**, whose truth table is shown below.

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
| TensorFlow | Build, train, and evaluate the neural network |
| NumPy | Store and manipulate numerical data |

---

## Model Architecture

```
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
- Learns hidden patterns from the data.

### Output Layer
- Contains **1 neuron**.
- Uses the **Sigmoid activation function**.
- Produces a probability between **0 and 1**.

---

## Algorithm

1. Import TensorFlow and NumPy.
2. Create the AND gate dataset.
3. Build a Sequential neural network.
4. Add one hidden layer with ReLU activation.
5. Add one output layer with Sigmoid activation.
6. Compile the model using:
   - Adam Optimizer
   - Binary Crossentropy Loss
   - Accuracy Metric
7. Display predictions before training.
8. Train the model for 10 epochs.
9. Predict outputs after training.
10. Compare the results.

---

## Code Explanation

### Step 1 — Import Libraries

Imports TensorFlow for building the neural network and NumPy for handling the dataset.

---

### Step 2 — Create Dataset

The AND gate dataset consists of four possible input combinations and their corresponding outputs.

---

### Step 3 — Build the Neural Network

A Sequential model is created with:

- Input Layer (2 features)
- Hidden Layer (4 neurons, ReLU)
- Output Layer (1 neuron, Sigmoid)

---

### Step 4 — Compile the Model

The model is compiled using:

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy

---

### Step 5 — Predictions Before Training

The model predicts outputs before learning.

Since the weights are randomly initialized, the predictions are essentially random.

---

### Step 6 — Train the Model

The model is trained for **10 epochs**.

During training, the network adjusts its weights to minimize prediction error.

---

### Step 7 — Predictions After Training

The trained model predicts the outputs again.

The sigmoid output is converted into class labels using a threshold of **0.5**.

- Output ≥ 0.5 → Class 1
- Output < 0.5 → Class 0

---

## Expected Output

### Before Training

The predictions are random because the model has not yet learned the relationship between the inputs and outputs.

Example:

```text
Input: [0 0] → Predicted: 0.61 → Class: 1
Input: [0 1] → Predicted: 0.44 → Class: 0
...
```

---

### After Training

The model learns the AND gate.

Example:

```text
Input: [0 0] → Predicted: 0.01 → Class: 0
Input: [0 1] → Predicted: 0.03 → Class: 0
Input: [1 0] → Predicted: 0.05 → Class: 0
Input: [1 1] → Predicted: 0.98 → Class: 1
```

---

## Result

The Feedforward Neural Network successfully learns the logical **AND** operation.

The predictions before training are random due to randomly initialized weights. After training, the model correctly classifies all input combinations, demonstrating how a neural network learns patterns from data through training.

---

## Applications

- Binary Classification
- Pattern Recognition
- Spam Detection
- Medical Diagnosis
- Credit Risk Prediction
- Fraud Detection

---

## Conclusion

This experiment demonstrates the implementation of a basic Feedforward Neural Network using TensorFlow. Although the dataset is very small, it illustrates the complete workflow of building, compiling, training, and evaluating a neural network. It also highlights the effect of training by comparing predictions before and after learning.

---

## References

1. TensorFlow Documentation — https://www.tensorflow.org/
2. NumPy Documentation — https://numpy.org/doc/
3. Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning*, MIT Press.