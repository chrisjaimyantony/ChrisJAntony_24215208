# Spam Classifier — PyTorch Feedforward Neural Network

A binary classification model built using **PyTorch** and **NumPy** that determines whether an email is **Spam** or **Legitimate** based on four measurable characteristics of the email. The project began as a simple neural network tutorial for an AND gate and was extended into a practical spam detection system.

---

# Project Overview

The model predicts whether an email is spam using four input features:

- Link Count
- Capital Letter Ratio
- Recipient Count
- Suspicious Word Count

Although the application differs greatly from a logic gate, the neural network architecture remains fundamentally the same.

---

# How It Works

The neural network accepts four numerical inputs, processes them through a hidden layer using the **ReLU activation function**, and produces a probability using a **sigmoid output neuron**.

Architecture:

```text
Input (4 Features)
        │
        ▼
Dense Layer (8 Neurons)
        │
        ▼
ReLU Activation
        │
        ▼
Dense Layer (1 Neuron)
        │
        ▼
Sigmoid
        │
        ▼
Spam or Legitimate
```

The sigmoid output produces a probability between **0 and 1**.

Classification rule:

```text
Probability ≥ 0.50  → Spam

Probability < 0.50  → Legitimate
```

Unlike TensorFlow's high-level training interface, PyTorch requires an explicit **forward()** function and manual training loop, making every stage of learning visible.

---

# PyTorch Training Process

Every training iteration follows the same sequence:

```text
Forward Pass
      │
      ▼
Compute Loss
      │
      ▼
Backpropagation
      │
      ▼
Optimizer Updates Weights
```

Typical PyTorch workflow:

```python
optimizer.zero_grad()

outputs = model(inputs)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

This explicit workflow provides complete control over training, allowing custom logging, debugging, or modifications at any stage.

---

# Project Workflow

| Step | Purpose |
|------|---------|
| Data Generation | Create synthetic spam and legitimate emails |
| Feature Normalization | Standardize every feature (mean = 0, standard deviation = 1) |
| Train/Test Split | Perform an 80/20 split before normalization to prevent data leakage |
| Model Training | Manual PyTorch training loop with loss and accuracy tracking |
| Model Evaluation | Accuracy, Precision, Recall, and Confusion Matrix |
| Prediction | Predict new emails with confidence scores |

---

# Neural Network Architecture

The model consists of:

```text
4 Input Features

↓

Dense Layer (8)

↓

ReLU

↓

Dense Layer (1)

↓

Sigmoid
```

The hidden layer extracts relationships between the input features.

The output neuron converts the learned representation into a probability that the email belongs to the spam class.

---

# Optimizer Comparison

Three optimizers were evaluated using:

- Identical architecture
- Same dataset
- 100 training epochs
- Learning rate = **0.01**

Only the optimization algorithm changed.

---

## Adam (Learning Rate = 0.01)

Results:

```text
Test Accuracy: 99.2%

Precision: 100.0%

Recall: 98.3%
```

Training characteristics:

- Rapid convergence
- Loss reduced from approximately **0.70** to **0.007**
- High confidence on obvious spam emails
- Reasonably confident on borderline examples

Adam combines adaptive learning rates with momentum, making it one of the most commonly used optimizers.

---

## SGD (Learning Rate = 0.01)

Results:

```text
Test Accuracy: 99.2%

Precision: 100.0%

Recall: 98.3%
```

Training characteristics:

- Much slower convergence
- Final loss approximately **0.333**
- Reached essentially the same decision boundary as Adam
- Required many more updates

SGD makes smaller, steadier parameter updates compared with Adam.

---

## RMSprop (Learning Rate = 0.01)

Results:

```text
Test Accuracy: 100.0%

Precision: 100.0%

Recall: 100.0%
```

Training characteristics:

- Fast convergence
- Final loss approximately **0.001**
- Classified every test email correctly
- Produced the strongest decision boundary on this dataset

Although the improvement over Adam was small, RMSprop achieved the best overall performance during this experiment.

---

# Optimizer Summary

| Optimizer | Test Accuracy | Final Loss | Borderline Email Confidence |
|------------|--------------:|-----------:|----------------------------:|
| Adam | 99.2% | 0.007 | 74.7% |
| SGD | 99.2% | 0.333 | Similar boundary to Adam |
| RMSprop | 100.0% | 0.001 | 77.8% |

For this clean, well-separated synthetic dataset:

- All optimizers converged successfully.
- RMSprop produced the lowest loss.
- Adam converged much faster than SGD.
- SGD eventually reached a similar solution but required considerably more training.

The differences become much larger when the data contains noise or overlapping classes.

---

# Learning Rate Experiment

To observe the effect of poor hyperparameter selection, Adam was retrained using an extremely small learning rate.

Learning rate:

```text
0.00001
```

Training output:

```text
Epoch 20

Loss: 0.7226

Training Accuracy: 14.6%


Epoch 100

Loss: 0.7212

Training Accuracy: 15.0%
```

Final result:

```text
Test Accuracy

16.7%
```

The model barely learned.

The loss remained close to **0.72**, which is approximately the loss expected from random guessing.

Rather than making incorrect updates, the optimizer simply moved too slowly.

Even after 100 epochs, the weights remained close to their initial random values.

This illustrates that an excessively small learning rate leads to **stagnation**, not instability.

---

# Predictions Before Training

Before training begins, each optimizer starts with randomly initialized weights.

Example outputs:

```text
                    Adam      SGD    RMSprop

Spam Email        0.5564   0.5148    0.4176

Spam Email        0.5438   0.5060    0.4185

Legitimate Email  0.5004   0.4986    0.4161
```

These predictions have no real meaning because the network has not yet learned any patterns.

The variation exists because every model begins with different random parameter values.

After training, all optimizers converge toward meaningful predictions.

---

# What Each Library Does

## NumPy

NumPy is responsible for all numerical processing outside the neural network.

Tasks include:

- Synthetic data generation
- Feature normalization
- Matrix operations
- Confusion matrix computation
- Performance evaluation
- Data preprocessing

---

## PyTorch

PyTorch performs the machine learning.

Responsibilities include:

- Defining the neural network using `nn.Module`
- Implementing the `forward()` method
- Automatic differentiation through `loss.backward()`
- Weight optimization using optimizers
- Prediction generation
- Training loop execution

Because PyTorch exposes every stage of the training process, it is particularly useful for experimentation and research.

---

# Key Takeaways

## 1. Learning Rate Is Critical

A poor learning rate can make an otherwise effective neural network appear completely ineffective.

With:

```text
Learning Rate = 0.00001
```

the network barely changed after 100 epochs.

With:

```text
Learning Rate = 0.01
```

all three optimizers converged successfully.

---

## 2. Optimizer Differences Become More Important on Difficult Problems

For clean datasets:

- Adam
- SGD
- RMSprop

all achieved nearly identical accuracy.

As datasets become noisier or more complex, optimizer choice has a much greater impact on convergence speed and final model performance.

---

## 3. PyTorch Provides Complete Transparency

Unlike higher-level frameworks, PyTorch exposes the entire learning process.

Developers can easily:

- Inspect gradients
- Monitor weight updates
- Print intermediate activations
- Modify loss functions
- Implement custom optimization techniques

This flexibility makes PyTorch especially popular in research and advanced machine learning applications.

---

# Future Improvements

Several experiments can further demonstrate the strengths and limitations of feedforward neural networks:

- Introduce overlapping spam and legitimate email distributions
- Reduce the training dataset size to observe overfitting
- Add validation monitoring during training
- Compare additional activation functions such as Leaky ReLU and ELU
- Experiment with deeper network architectures and dropout regularization

These extensions provide valuable insight into model generalization, optimization, and robustness.

---

# Dependencies

- **PyTorch** – Neural network definition, automatic differentiation, optimization, and training
- **NumPy** – Data generation, preprocessing, normalization, and model evaluation