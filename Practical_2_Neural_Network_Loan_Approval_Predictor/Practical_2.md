# Neural Network Loan Approval Predictor

A binary classification model built using **TensorFlow/Keras** and **NumPy** that predicts whether a loan application should be approved based on financial information. The project began as a simple AND gate neural network tutorial and was progressively expanded into a practical machine learning application.

---

# Project Overview

The objective is to predict whether a loan applicant should be **Approved** or **Rejected** using four financial features:

- Monthly Income
- Credit Score
- Debt-to-Income Ratio
- Years of Employment

Although the problem is much more realistic than a logic gate, the underlying neural network architecture remains fundamentally the same.

---

# How It Started

The original project was the classic TensorFlow implementation of an **AND gate**, one of the simplest binary classification problems.

```python
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0],   [0],   [0],   [1]]

model = Sequential([
    Dense(4, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid")
])
```

The network consists of:

- An input layer
- One hidden layer
- A sigmoid output neuron

The sigmoid output produces a probability between **0 and 1**, making the architecture suitable for **any binary classification task**, not just logic gates.

Changing the dataset transforms the problem from digital logic into loan approval without changing the overall network structure.

---

# From Tutorial to Real Application

Several components were added to make the model resemble a practical machine learning pipeline.

| Aspect | AND Gate | Loan Approval |
|---------|----------|---------------|
| Input Features | 2 binary inputs | 4 financial features |
| Dataset Size | 4 samples | 400 samples |
| Hidden Neurons | 4 | 8 |
| Feature Scaling | Not required | Standardization required |
| Output | Binary prediction | Prediction with confidence score |
| Evaluation | Printed outputs | Accuracy, Precision, Recall, Confusion Matrix |

The neural network itself changed very little.

Most of the improvements came from everything surrounding the model:

- Better data generation
- Feature normalization
- Model evaluation
- Real-world prediction examples
- Confidence score interpretation

---

# Neural Network Architecture

The final model consists of:

```text
Input Layer (4 Features)
        │
        ▼
Dense Layer (8 Neurons, ReLU)
        │
        ▼
Dense Layer (1 Neuron, Sigmoid)
```

### Hidden Layer

The hidden layer uses the **ReLU activation function**.

ReLU outputs:

```text
max(0, x)
```

which allows the network to learn increasingly complex feature relationships while avoiding many of the training issues associated with older activation functions.

### Output Layer

The final neuron uses the **sigmoid activation function**.

```text
σ(x) = 1 / (1 + e⁻ˣ)
```

The output is interpreted as the probability of loan approval.

Example:

```text
0.98 → 98% chance of approval
0.15 → 15% chance of approval
```

---

# Data Preparation

Unlike logic gates, financial features exist on completely different scales.

Example:

| Feature | Example Value |
|----------|--------------:|
| Income | 6500 |
| Credit Score | 720 |
| Debt Ratio | 0.35 |
| Years Employed | 8 |

Without scaling, the larger numerical values dominate gradient updates.

Therefore each feature is standardized using:

```text
z = (x − mean) / standard deviation
```

This allows every feature to contribute proportionally during training.

---

# Experiments

Several experiments were performed to understand how different design choices affect model behavior.

---

## Optimizer Comparison

Three optimizers were tested.

- Adam
- SGD
- RMSprop

All eventually learned similar decision boundaries on clean data.

The primary difference appeared in borderline cases.

Example applicant:

```text
Income:        5500
Credit Score:   690
Debt Ratio:    0.30
Employment:      5 years
```

Predicted confidence:

```text
Adam      → 95.1%

RMSprop   → 76.3%

SGD        → 60.3%
```

### Interpretation

Adam produced the most confident predictions.

SGD generated more conservative probabilities.

RMSprop behaved somewhere between the two.

On clean datasets this difference is relatively unimportant.

On noisy datasets, however, Adam's higher confidence can sometimes lead to overconfident incorrect predictions.

---

## Activation Function Comparison

Two hidden-layer activation functions were compared.

- ReLU
- tanh

Before training, their outputs looked very different.

### ReLU

Predictions clustered around:

```text
0.5
```

This represents a neutral starting point before learning.

### tanh

Predictions were spread further apart and initially appeared inverted.

Approved applicants sometimes received lower probabilities than rejected applicants.

This happens because:

```text
tanh(x)

outputs values between

-1 and 1
```

whereas

```text
ReLU(x)

outputs

0 to ∞
```

These different activation ranges produce different initial network behavior.

After training, both activation functions converged to correct predictions.

Only the starting point differed.

---

## Learning Rate

Two extreme learning rates were tested.

### Very Small Learning Rate

```text
0.0001
```

Training became extremely slow.

Accuracy improved only gradually and required many more epochs.

---

### Very Large Learning Rate

```text
0.1
```

Loss oscillated instead of decreasing smoothly.

The optimizer repeatedly overshot the optimum.

---

## Adding Another Hidden Layer

A second hidden layer was introduced.

Result:

Very little improvement.

The synthetic dataset was already close to linearly separable, so additional network depth added unnecessary complexity.

---

## Dropout

A dropout layer with probability:

```text
0.5
```

was added.

Effects:

- Lower training accuracy
- More stable validation performance
- Better resistance to overfitting

Dropout randomly disables neurons during training, preventing the network from relying too heavily on specific pathways.

---

## Overlapping Classes

The PASS and FAIL distributions were intentionally moved closer together.

Effects:

- Test accuracy dropped to roughly **70–85%**
- Confidence scores became much less certain
- Borderline applicants became difficult to classify

This better reflects real-world financial data, where classes rarely separate perfectly.

---

## Small Dataset

Training was repeated with only:

```text
40 samples
```

Results:

- Nearly 100% training accuracy
- Highly unstable test accuracy

The network memorized the training data instead of learning general patterns.

---

## Validation Monitoring

Validation accuracy was tracked during training.

This revealed the difference between:

- learning
- memorization

One experiment combined:

- overlapping classes
- small dataset
- validation monitoring

Results:

```text
Training Accuracy

≈ 100%

Validation Accuracy

≈ 60–75%
```

This large gap clearly indicates **overfitting**.

Without validation monitoring, the model would appear almost perfect despite performing poorly on unseen data.

---

# What Each Library Does

## NumPy

NumPy handles every stage outside the neural network itself.

Responsibilities include:

- Generating synthetic datasets
- Feature normalization
- Data preprocessing
- Confusion matrix calculations
- Converting normalized values back into readable units
- General numerical computation

---

## TensorFlow / Keras

TensorFlow performs the machine learning.

Responsibilities include:

- Building the neural network
- Automatic differentiation
- Gradient computation
- Weight updates
- Optimizer implementation
- Model training
- Prediction generation

Implementing these algorithms manually would require substantial mathematical and programming effort.

---

# Key Takeaways

## 1. Neural Networks Are General-Purpose Models

The same architecture that solves an AND gate can also solve financial classification problems.

Only the dataset changes.

---

## 2. Optimizers Matter Most on Difficult Problems

For simple datasets:

- Adam
- SGD
- RMSprop

all converge to similar solutions.

As data becomes noisier, optimizer choice begins affecting confidence and convergence behavior.

---

## 3. Confidence Is More Informative Than the Class Label

Two predictions may both say:

```text
Approved
```

yet have very different meanings.

```text
52% confidence

vs

99% confidence
```

Confidence provides valuable information about prediction certainty.

---

## 4. Validation Monitoring Is Essential

Training accuracy alone is insufficient.

A large gap between training and validation performance usually indicates overfitting.

Monitoring both provides a much clearer picture of model quality.

---

## Future Experiment

The next planned improvement is to intentionally create conditions that encourage overfitting.

The experiment combines:

- A very small training dataset
- Overlapping class distributions
- Validation monitoring during training

This setup should clearly demonstrate how neural networks can memorize training data while failing to generalize to unseen examples, making it an effective illustration of overfitting.

---

# Dependencies

- **TensorFlow / Keras** – Neural network construction, optimization, and training
- **NumPy** – Data generation, preprocessing, normalization, and evaluation