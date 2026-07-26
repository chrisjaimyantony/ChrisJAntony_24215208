# Learning Rules and MLP Auto-Comparison

A comparative machine learning study that evaluates five classical single-neuron learning rules against a **TensorFlow/Keras Multi-Layer Perceptron (MLP)**. Every model is trained and tested on the same dataset, allowing direct comparisons of learning behavior, convergence, and generalization.

Rather than comparing different datasets, this project compares **how different learning algorithms solve the exact same problem**, making it an excellent demonstration of the evolution from classical neural learning rules to modern deep learning.

---

# Project Overview

The program is divided into four sequential experiments.

## Part 1 — Classical Learning Rules

A single neuron is trained using five different learning rules.

Every rule uses:

- The same dataset
- The same initial weights
- The same learning rate
- The same number of epochs

This ensures the only difference is **how the weights are updated**.

---

## Part 2a — Activation Function Sweep

A Keras Multi-Layer Perceptron automatically trains every combination of:

- 4 activation functions
- 4 learning rates

Result:

```text
4 Activations × 4 Learning Rates

=

16 Neural Networks
```

Each model is ranked by its test accuracy and train-test generalization gap.

---

## Part 2b — Optimizer Sweep

Using the best activation function from Part 2a, the program compares:

- Adam
- SGD
- RMSprop

across three learning rates.

Result:

```text
3 Optimizers

×

3 Learning Rates

=

9 Models
```

The highest-performing configuration is selected automatically.

---

## Part 3 — New Predictions

The winning neural network is retrained and evaluated on previously unseen examples.

Each prediction includes:

- Predicted class
- Confidence score

---

## Part 4 — Final Comparison

The project concludes by comparing:

- The best classical learning rule
- The best Multi-Layer Perceptron

This provides a direct comparison between traditional single-neuron learning and modern neural networks.

---

# The Five Learning Rules

Each learning rule modifies weights differently.

---

## 1. Hebbian Learning

Learning principle:

> Neurons that activate together strengthen their connection.

Weight updates depend only on the correlation between input and output.

Characteristics:

- Unsupervised
- No error signal
- Strengthens existing associations

Major limitation:

Incorrect weight updates cannot be reversed because the rule has no concept of prediction error.

---

## 2. Perceptron Learning

The perceptron updates weights only when a prediction is incorrect.

Learning process:

```text
Prediction

↓

Compare with Target

↓

If Wrong

↓

Update Weights
```

Characteristics:

- Supervised
- Error-driven
- Uses a binary step activation

Major limitation:

The step function provides no gradient information, so updates are coarse rather than smooth.

---

## 3. Delta Rule

The Delta Rule replaces the step function with a differentiable sigmoid function.

Weight updates become proportional to both:

- Prediction error
- Sigmoid gradient

Characteristics:

- Supervised
- Gradient-based
- Smooth learning

Major limitation:

Requires a differentiable activation function.

---

## 4. Correlation Learning

Correlation Learning is essentially supervised Hebbian learning.

Instead of reinforcing the neuron's prediction, it reinforces the target label directly.

Characteristics:

- Supervised
- Correlation-based
- Includes weight decay to prevent runaway growth

Major limitation:

It amplifies desired outputs but still lacks true error correction.

---

## 5. Outstar Learning

Outstar learning directly moves each weight toward its desired value.

Characteristics:

- Supervised
- Direct weight adjustment
- Fast convergence

Major limitation:

Each weight is treated independently rather than considering the complete decision boundary.

---

# Results — Classical Learning Rules

| Learning Rule | Training Accuracy | Test Accuracy |
|---------------|-----------------:|--------------:|
| Hebbian | 80.6% | 82.5% |
| Perceptron | 92.2% | 87.5% |
| Delta | 95.9% | 90.0% |
| Correlation | 83.4% | 82.5% |
| Outstar | 95.6% | 90.0% |

### Observations

Delta and Outstar produced the strongest performance.

Both achieved:

```text
90.0% Test Accuracy
```

Their advantage comes from using continuous learning signals instead of discrete updates.

The Perceptron performed well but lost some information because every prediction is reduced to either:

```text
0

or

1
```

before learning.

---

## Hebbian and Correlation Learning

Both learning rules plateaued very early.

Training accuracy stopped improving long before the maximum number of epochs.

Reason:

Neither algorithm contains an error-correction mechanism.

They continually reinforce relationships but cannot actively repair incorrect decisions.

---

# Results — Activation Function Sweep

Sixteen neural networks were trained automatically.

| Activation | Learning Rate | Train Accuracy | Test Accuracy | Train-Test Gap |
|------------|--------------:|---------------:|--------------:|---------------:|
| ReLU | 0.001 | 95.6% | 91.3% | 4.4% |
| tanh | 0.010 | 95.6% | 91.3% | 4.4% |
| Sigmoid | 0.010 | 96.2% | 91.3% | 5.0% |
| ELU | 0.100 | 95.3% | 91.3% | 4.1% |

---

## Observations

All four activation functions eventually reached essentially the same performance.

Maximum test accuracy:

```text
91.3%
```

This happened because the dataset was already almost perfectly linearly separable.

The activation function therefore had relatively little influence.

---

### Sigmoid Failure

One configuration performed dramatically worse.

Configuration:

```text
Activation:

Sigmoid

Learning Rate:

0.0001
```

Training accuracy:

```text
49.1%
```

approximately random guessing.

Reason:

Sigmoid compresses activations into a narrow range.

Combined with an extremely small learning rate, gradients became too small to produce meaningful weight updates.

This is a small-scale demonstration of the **vanishing gradient problem**.

---

# Optimizer Comparison

Nine additional models compared three optimizers.

| Optimizer | Learning Rate | Train Accuracy | Test Accuracy | Gap |
|------------|--------------:|---------------:|--------------:|----:|
| Adam | 0.100 | 96.6% | 91.3% | 5.3% |
| SGD | 0.010 | 94.7% | 91.3% | 3.4% |
| RMSprop | 0.001 | 95.3% | 91.3% | 4.1% |

---

## Observations

All optimizers achieved identical test accuracy.

The difference appeared in the train-test gap.

SGD produced:

```text
Smallest Gap

3.4%
```

meaning it generalized more consistently.

Adam achieved higher training accuracy but no improvement in testing performance.

The additional training performance represented memorization rather than better generalization.

---

# Single Neuron vs Multi-Layer Perceptron

Final comparison:

```text
Best Classical Rule

Delta

90.0%
```

```text
Best Neural Network

Adam + ReLU

91.3%
```

Improvement:

```text
+1.3%
```

---

## Why Was the Improvement So Small?

The dataset was almost perfectly linearly separable.

A single neuron already learns a straight decision boundary.

Adding hidden layers provides little additional benefit because there are no complex feature interactions to learn.

---

## When Would the MLP Win by More?

Suppose risk depends on multiple interacting features.

Example:

```text
Young Age

↓

Normally Low Risk

BUT

High BMI

+

High Blood Sugar

↓

High Risk
```

No straight line can represent this interaction.

A Multi-Layer Perceptron can learn these nonlinear relationships.

A single neuron cannot.

---

# Predictions on New Data

| Feature 1 | Feature 2 | Confidence | Predicted Class |
|-----------:|----------:|-----------:|----------------:|
| 2.0 | 50.0 | 100.0% | Class 0 |
| 8.0 | 85.0 | 100.0% | Class 1 |
| 5.0 | 65.0 | 92.1% | Class 1 |
| 6.5 | 75.0 | 99.7% | Class 1 |
| 3.0 | 48.0 | 100.0% | Class 0 |
| 9.0 | 90.0 | 100.0% | Class 1 |

Clear examples received near-certain predictions.

Borderline cases received lower confidence, reflecting greater uncertainty.

This behavior is desirable because confidence should reflect how ambiguous an example is rather than always producing extreme probabilities.

---

# What Each Library Does

## NumPy

NumPy implements the classical machine learning algorithms.

Responsibilities include:

- All five learning rules
- Synthetic data generation
- Feature normalization
- Weight updates
- Evaluation metrics
- Numerical computation

---

## TensorFlow / Keras

TensorFlow/Keras handles the Multi-Layer Perceptron.

Responsibilities include:

- Network architecture
- Backpropagation
- Automatic differentiation
- Optimizer management
- Model training
- Prediction generation

While the single-neuron learning rules are compact enough to implement manually, backpropagation through multiple hidden layers would require substantially more code and mathematical bookkeeping, making Keras the practical choice.

---

# Key Takeaways

## 1. Learning Rule Matters More Than Architecture on Simple Problems

For a linearly separable dataset:

```text
Single Neuron

90.0%

vs

MLP

91.3%
```

The additional network complexity provides only a modest improvement.

---

## 2. Unsupervised Learning Has Natural Limits

Hebbian learning successfully discovers correlations but cannot perform reliable classification because it lacks an error-correction mechanism.

It is better suited for representation learning than decision making.

---

## 3. Learning Rate Has Greater Impact Than Optimizer Choice

Example:

```text
SGD

Learning Rate = 0.001

↓

47.5%
```

```text
SGD

Learning Rate = 0.01

↓

91.3%
```

The optimizer remained the same.

Only the learning rate changed.

This demonstrates that selecting an appropriate learning rate is often more important than selecting the optimizer itself.

---

## 4. Many Configurations Perform Equally Well on Simple Data

Clean datasets often allow many combinations of:

- activation functions
- optimizers
- learning rates

to achieve nearly identical results.

The differences become much more pronounced on realistic datasets containing nonlinear relationships, overlapping classes, or noise.

---

## 5. Always Monitor the Train-Test Gap

Training accuracy alone does not indicate good generalization.

For example:

```text
94% Train

91% Test
```

is generally preferable to:

```text
97% Train

89% Test
```

because the smaller gap indicates the model is learning patterns rather than memorizing the training data.

---

# Future Improvements

The next logical extension is to introduce **nonlinear feature interactions** into the dataset.

For example, a medical-risk scenario where individual features appear low risk in isolation but become high risk when combined (similar to metabolic syndrome patterns).

Such a dataset would:

- Challenge the single-neuron learning rules
- Increase the advantage of the Multi-Layer Perceptron
- Better demonstrate why hidden layers are necessary for modeling complex decision boundaries

This experiment would highlight the transition from problems that are solvable with linear models to those that require the representational power of modern neural networks.

---

# Dependencies

- **NumPy** – Classical learning rules, numerical computation, data generation, preprocessing, and evaluation
- **TensorFlow / Keras** – Multi-Layer Perceptron implementation, backpropagation, optimizers, automatic differentiation, and training