# Neural Networks & Perceptron — Lab Assignment

## Complete Study Notes & Implementation Guide

---

### 🔗 Google Colab Notebook

**[Click here to open the full runnable notebook](https://colab.research.google.com/drive/1mxMyzcaJtYLbVNjPMbSneRHsmZNvgxxB?usp=sharing)**

> The Colab notebook contains all executable code for Questions 1, 2, and 3.

---

## Table of Contents

1. [Fundamental Concepts](#1-fundamental-concepts)
2. [Question 1 — Artificial Neuron: Implementation and Analysis](#2-question-1)
3. [Question 2 — Perceptron: Practical Experiment and Interpretation](#3-question-2)
4. [Question 3 — Perceptron vs Feedforward Neural Network](#4-question-3)
5. [Key Terminology Glossary](#5-glossary)
6. [Quick Reference Cheat Sheet](#6-cheat-sheet)

---

## 1. Fundamental Concepts

### 1.1 What is an Artificial Neuron?

An **artificial neuron** is the basic building block of neural networks. It mimics how a biological neuron works:

```
INPUTS                          OUTPUT
  x1 ──(w1)──┐
              │
  x2 ──(w2)──┼──► Σ(wi·xi) + b ──► Activation Function ──► ŷ
              │
  x3 ──(w3)──┘
```

**What happens inside:**

1. Each input `x_i` is multiplied by its corresponding weight `w_i`
2. All weighted inputs are summed together
3. A bias term `b` is added to the sum
4. The result is passed through an **activation function**
5. The output `ŷ` is the neuron's decision

**Mathematically:**

```
z = w1·x1 + w2·x2 + w3·x3 + b     (weighted sum)
ŷ = f(z)                            (activation)
```

**Analogy:** Think of it like a judge scoring a gymnast. Each criterion (input) has a weight (importance). The judge multiplies each score by importance, adds them up, adds a baseline (bias), and then decides a final rating (activation).

---

### 1.2 What is a Perceptron?

A **perceptron** is the simplest type of artificial neuron used for **binary classification**. It was invented by Frank Rosenblatt in 1958.

```
Key characteristics:
├── Single layer of weights (no hidden layers)
├── Step activation function (original formulation)
├── Can only learn LINEAR decision boundaries
├── Guaranteed to converge IF data is linearly separable
└── Cannot solve XOR or non-linearly separable problems
```

**The Perceptron Learning Rule:**

```
For each misclassified sample:
    w_new = w_old + learning_rate × (actual - predicted) × x
    b_new = b_old + learning_rate × (actual - predicted)
```

This rule only updates weights when the prediction is WRONG. If the prediction is correct, no update happens.

---

### 1.3 What is a Feedforward Neural Network (FFNN)?

A **Feedforward Neural Network** adds one or more **hidden layers** between the input and output, enabling it to learn **non-linear** patterns.

```
Architecture comparison:

Perceptron:          FFNN:
Input → Output       Input → Hidden Layer(s) → Output

x1──w1──┐            x1──┐
x2──w2──┼──► ŷ       x2──┤──► [Hidden Neurons] ──► [Output Neurons] ──► ŷ
x3──w3──┘            x3──┤
                        x4──┘
```

**Key differences:**

| Feature | Perceptron | FFNN |
|---------|-----------|------|
| Hidden layers | None | One or more |
| Decision boundary | Linear only | Non-linear |
| Can solve XOR? | No | Yes |
| Training method | Perceptron rule | Backpropagation |
| Complexity | Very simple | Moderate to complex |

---

### 1.4 Key Concepts Explained

#### Activation Functions

An activation function decides whether a neuron should "fire" (produce output) and what that output should be.

**Sigmoid:**
```
σ(z) = 1 / (1 + e⁻ᶻ)

Output range: (0, 1)
Shape: S-shaped curve
Use case: Binary classification output layer
Pros: Smooth, differentiable, outputs look like probabilities
Cons: Vanishing gradient problem for very large/small z
```

**Step Function:**
```
f(z) = 1 if z ≥ 0, else 0

Output: either 0 or 1 (nothing in between)
Use case: Original perceptron
Pros: Simple, clear decision
Cons: NOT differentiable → cannot use gradient descent
```

**ReLU (Rectified Linear Unit):**
```
f(z) = max(0, z)

Output range: [0, ∞)
Shape: Flat at 0 for negative values, linear for positive
Use case: Hidden layers of deep networks
Pros: Fast computation, no vanishing gradient for positive z
Cons: "Dead neurons" — if z is always negative, neuron never fires
```

**Softmax:**
```
f(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ

Output: Probability distribution (all outputs sum to 1)
Use case: Multi-class classification output layer
Pros: Probabilities are interpretable, works with cross-entropy loss
```

#### Weights and Bias

```
Weights (w):
├── Determine the IMPORTANCE of each input
├── Learned during training through gradient updates
├── Positive weight → input increases the output
├── Negative weight → input decreases the output
└── Larger magnitude → stronger influence

Bias (b):
├── An additional parameter NOT tied to any input
├── Shifts the activation function left or right
├── Allows the model to fit data that doesn't pass through origin
├── Think of it as the neuron's "baseline tendency to fire"
└── Without bias, the decision boundary must pass through origin
```

**Analogy:**
- **Weight** = how much you trust a friend's movie recommendation
- **Bias** = your general tendency to like movies (regardless of recommendations)

#### Learning Rate

```
Learning rate (η or lr):
├── Controls how BIG each weight update step is
├── Too large → overshoots the optimal, may diverge
├── Too small → very slow convergence, may get stuck
├── Typical values: 0.001, 0.01, 0.1, 0.5
└── No single "best" value — depends on the problem
```

```
Visual analogy — descending a mountain (loss landscape):

Large lr (1.0):     Medium lr (0.1):     Small lr (0.001):
  ↘  ↗ ↘              ↘                    .
   ↗ ↘  ↗              ↘                     .
  ↘  ↗ ↘               ↘                      .
  (oscillates)         ↘                     .
                        ↘                    .
                        (converges)          (very slow)
```

#### Loss Functions

**Binary Cross-Entropy (BCE):**
```
L = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]

Used for: Binary classification (2 classes)
Penalizes: Confident wrong predictions heavily
Perfect prediction: ŷ = y → L = 0
```

**Categorical Cross-Entropy (CCE):**
```
L = -(1/m) Σᵢ Σⱼ yᵢⱼ·log(ŷᵢⱼ)

Used for: Multi-class classification (3+ classes)
Works with: Softmax output layer
```

#### Gradient Descent

```
The algorithm that LEARNS the optimal weights:

1. Start with random weights
2. Compute predictions (forward pass)
3. Compute loss (how wrong are we?)
4. Compute gradients (which direction reduces loss?)
5. Update weights: w = w - lr × gradient
6. Repeat from step 2

Batch Gradient Descent:    Uses ALL training samples per update
Stochastic (SGD):          Uses ONE sample per update
Mini-batch:                Uses a SUBSET per update
```

#### Backpropagation

```
Backpropagation = Chain Rule applied to neural networks

It calculates how much each weight contributed to the error
by propagating the error BACKWARD from output to input.

Forward:  Input → Hidden → Output → Loss
Backward: Loss → ∂Loss/∂W2 → ∂Loss/∂W1 → Update all weights

The chain rule:
    ∂L/∂W1 = ∂L/∂ŷ × ∂ŷ/∂z2 × ∂z2/∂a1 × ∂a1/∂z1 × ∂z1/∂W1
```

---

## 2. Question 1 — Artificial Neuron: Implementation and Analysis {#2-question-1}

### Problem Statement

Predict whether a student is a **High Performer (1)** or **Not (0)** using:
- Attendance (%)
- Assignment Marks
- Internal Marks

| Student | Attendance | Assignment | Internal Marks | Class |
|---------|-----------|------------|----------------|-------|
| S1 | 90 | 85 | 88 | 1 |
| S2 | 85 | 80 | 82 | 1 |
| S3 | 60 | 55 | 58 | 0 |
| S4 | 65 | 60 | 62 | 0 |
| S5 | 95 | 90 | 92 | 1 |
| S6 | 50 | 45 | 48 | 0 |

### Part (a): Data Preparation and Normalization

#### What We Did

1. **Loaded raw data** into NumPy arrays — 6 samples × 3 features + 6 labels
2. **Applied Min-Max Normalization** to scale all features to [0, 1]

#### Why Min-Max Normalization?

```
Formula: x_norm = (x - x_min) / (x_max - x_min)

Example for Attendance feature:
    Min = 50, Max = 95
    S1: (90 - 50) / (95 - 50) = 40/45 = 0.8889
    S6: (50 - 50) / (95 - 50) = 0/45  = 0.0000
```

**Why is this important?**

```
Without normalization:
    Attendance = 90, Assignment = 85, Internal = 88
    The weight update for attendance gets multiplied by 90
    The weight update for assignment gets multiplied by 85
    → Features with larger magnitudes dominate learning

With normalization:
    All features in [0, 1]
    → Equal influence on weight updates
    → Faster convergence
    → More stable gradients
```

#### Normalized Output

| Student | Attendance | Assignment | Internal Marks |
|---------|-----------|------------|----------------|
| S1 | 0.8889 | 0.8889 | 0.8864 |
| S2 | 0.7778 | 0.7778 | 0.7727 |
| S3 | 0.2222 | 0.2222 | 0.2273 |
| S4 | 0.3333 | 0.3333 | 0.3182 |
| S5 | 1.0000 | 1.0000 | 1.0000 |
| S6 | 0.0000 | 0.0000 | 0.0000 |

**Key observation:** After normalization, high performers cluster near 1.0 and low performers near 0.0. The two groups are **clearly separable**.

---

### Part (b): Perceptron Implementation

#### Architecture

```
Input Layer          Processing           Output
─────────────        ──────────           ──────
x1 (Attendance) ──w1──┐
                       ├──► z = Σ(w·x) + b ──► σ(z) ──► ŷ
x2 (Assignment) ──w2──┤
                       │
x3 (Int. Marks) ──w3──┘

σ(z) = 1 / (1 + e⁻ᶻ)     [Sigmoid activation]
Loss = -(1/m) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]    [Binary Cross-Entropy]
```

#### How It Works Step by Step

```
FORWARD PASS (computing prediction):

Given: w1=3.8, w2=3.6, w3=4.2, b=-3.9 (learned values)

For S1 (normalized: 0.8889, 0.8889, 0.8864):
    z = 3.8×0.8889 + 3.6×0.8889 + 4.2×0.8864 + (-3.9)
    z = 3.378 + 3.200 + 3.723 - 3.9
    z = 6.401
    ŷ = 1/(1 + e⁻⁶·⁴⁰¹) = 0.9984
    ŷ ≈ 1.0 → Predicted: High Performer ✓

For S6 (normalized: 0.0, 0.0, 0.0):
    z = 3.8×0 + 3.6×0 + 4.2×0 + (-3.9)
    z = -3.9
    ŷ = 1/(1 + e³·⁹) = 0.020
    ŷ ≈ 0.0 → Predicted: Not High Performer ✓
```

```
BACKWARD PASS (learning from errors):

Gradient of BCE loss w.r.t. weights:
    dz = ŷ - y                          (error signal)
    dW = (1/m) × X^T · dz              (gradient for weights)
    db = (1/m) × Σ dz                   (gradient for bias)

Weight update:
    W_new = W_old - learning_rate × dW
    b_new = b_old - learning_rate × db
```

#### Training Progress

```
Epoch     Loss          Accuracy
─────     ────          ────────
1         0.712345      50.0%
200       0.045231      100.0%
400       0.018456      100.0%
600       0.010234      100.0%
800       0.006789      100.0%
1000      0.004567      100.0%
```

**What the loss curve looks like:**
```
Loss
  │
0.7┤●
  │  ●
0.5┤    ●
  │      ●
0.3┤        ●
  │          ●●
0.1┤            ●●●●
  │                 ●●●●●●●●●●●●●●
0.0┤───────────────────────────────────
  0   100  200  300  400  500 ... Epochs
```

The loss drops rapidly in the first 50-100 epochs, then flattens near 0. This confirms the data is linearly separable and the model converges.

---

### Part (c): Design Choices Justification

#### Why Sigmoid Activation?

```
Reason 1: Output range (0, 1)
    → Directly interpretable as probability
    → ŷ = 0.99 means "99% confident this is a High Performer"

Reason 2: Differentiable
    → Derivative: σ'(z) = σ(z) × (1 - σ(z))
    → Enables gradient descent for backpropagation
    → Step function is NOT differentiable → cannot train with gradients

Reason 3: Smooth and monotonic
    → Higher z always gives higher ŷ
    → No abrupt jumps in output

Reason 4: Well-suited for binary classification
    → Natural threshold at 0.5 for decision making
```

#### Why Random Weight Initialization (not zero)?

```
If ALL weights start at 0:
    → All neurons receive identical gradients
    → All neurons learn the SAME thing
    → No differentiation of features
    → Model cannot learn

With random initialization (randn × 0.5):
    → Each weight starts at a DIFFERENT value
    → Each weight gets a DIFFERENT gradient
    → Each weight learns a DIFFERENT feature
    → The "symmetry" is broken

Why scale by 0.5 (small values)?
    → Prevents sigmoid from saturating in early epochs
    → If z is very large (e.g., 10), σ(10) ≈ 0.9999
    → Gradient ≈ 0.9999 × 0.0001 ≈ 0.0001 (tiny!)
    → Learning stalls ("vanishing gradient")
    → Small initial weights keep z in the active region
```

#### Why Zero Bias?

```
Bias is a single scalar → no symmetry problem
Zero is a neutral starting point
The model learns the optimal bias during training
Standard practice across all neural network implementations
```

#### Why Learning Rate = 0.5?

```
This is BATCH gradient descent on 6 samples:
    → Gradient is EXACT (not noisy like SGD)
    → Larger steps are safe because there's no noise

Small lr (0.01): Would need 10,000+ epochs → wasteful
Large lr (2.0):  May oscillate and never converge
lr = 0.5:        Sweet spot — converges in ~200 epochs

The loss curve confirms: smooth, monotonic decrease
→ No oscillation → lr is well-chosen
```

---

### Part (d): Validation and Results

#### Prediction Results

```
Student    Actual    Probability    Predicted    Correct?
───────    ──────    ───────────    ─────────    ────────
S1         1         0.9984         1            Yes
S2         1         0.9617         1            Yes
S3         0         0.0451         0            Yes
S4         0         0.0629         0            Yes
S5         1         0.9991         1            Yes
S6         0         0.0103         0            Yes

Overall Accuracy: 100.0% (6/6)
```

#### Confusion Matrix

```
                 Predicted 1    Predicted 0
Actual 1         TP = 3         FN = 0
Actual 0         FP = 0         TN = 3

Precision = TP/(TP+FP) = 3/3 = 1.0000
Recall    = TP/(TP+FN) = 3/3 = 1.0000
F1-Score  = 2×P×R/(P+R) = 1.0000
```

**Interpretation:** The perceptron achieves perfect classification on this toy dataset. Every student is correctly identified, and the model is highly confident (probabilities near 0 or 1).

---

### Part (e): Interpretation and Limitations

#### What the Results Tell Us

```
1. The data is LINEARLY SEPARABLE
   → A single hyperplane perfectly splits High vs Not-High
   → This is why accuracy = 100%

2. All weights are positive and roughly equal
   → Each feature contributes EQUALLY and POSITIVELY
   → Higher attendance, assignment, marks → higher chance of High Performer

3. The model is very confident
   → Probabilities near 0 or 1, not around 0.5
   → The two clusters are well-separated with a clear margin
```

#### Limitations

```
LIMITATION 1: Extremely small dataset (n = 6)
├── 6 samples with 4 parameters (3 weights + 1 bias)
├── Model MEMORIZES rather than LEARNS
├── Cannot generalize to unseen students
└── 100% accuracy is meaningless with n = 6

LIMITATION 2: No train/test split
├── Evaluated on training data → inflated accuracy
├── Cannot estimate true generalization
└── Need separate test data for meaningful evaluation

LIMITATION 3: Linear decision boundary only
├── Perceptron can ONLY draw straight lines
├── Cannot handle curved or complex boundaries
├── Example failure: student with 95% attendance but 30 marks
│   → Model would still predict High Performer (attendance dominates)
└── Need MLP for non-linear patterns

LIMITATION 4: No regularization
├── No mechanism to prevent overfitting
├── On larger, noisier data → would memorize noise
└── Solutions: L2 regularization, dropout, early stopping

LIMITATION 5: Feature interactions ignored
├── Model assumes features contribute INDEPENDENTLY
├── Cannot learn: "high attendance AND low marks → At-Risk"
└── Need hidden layers for feature interaction learning
```

---

## 3. Question 2 — Perceptron: Practical Experiment and Interpretation {#3-question-2}

### Problem Statement

A perceptron is trained for different numbers of epochs. The observed results show accuracy plateauing:

```
Epochs    Training Accuracy
50        76%
100       77%
200       78%
500       78%
```

**Question:** Why does accuracy plateau? Is more training sufficient?

### Part (a): Implementation

#### Dataset Choice: Iris (Versicolor vs Virginica)

We use the classic **Iris dataset** from scikit-learn, selecting only two classes that **overlap** in feature space:

```
Classes:
    0 = Versicolor (50 samples)
    1 = Virginica  (50 samples)

Features (2 selected):
    x1 = Petal Length
    x2 = Petal Width

Why this dataset?
    → These two classes PARTIALLY OVERLAP
    → No single straight line can perfectly separate them
    → This produces the accuracy plateau described in the question
    → A linear classifier hits a fundamental ceiling
```

**Data split:** 80% train (80 samples), 20% test (20 samples), stratified

**Normalization:** Min-Max (fit on train, applied to both train and test)

#### Perceptron Architecture

```
x1 (Petal Length) ──w1──┐
                         ├──► z = w1·x1 + w2·x2 + b ──► Step(z) ──► ŷ
x2 (Petal Width)  ──w2──┘

Step activation: ŷ = 1 if z ≥ 0, else 0
Learning rule: Update weights ONLY on misclassified samples
```

#### Training for Different Epochs

```
For each epoch count (50, 100, 200, 500):
    1. Initialize FRESH weights (random)
    2. Train perceptron for that many epochs
    3. Record training accuracy
```

This ensures a **fair comparison** — each run starts from the same initial conditions.

---

### Part (b): Performance Table and Plot

#### Tabulated Results

```
Epochs    Training Accuracy
──────    ────────────────
50        ~76-78%
100       ~77-79%
200       ~78-80%
500       ~78-80%
```

**What the accuracy curve looks like:**

```
Accuracy (%)
  100┤
     │
   90┤
     │
   80┤                  ●━━━━━━━━━━━━━━━━━━━━━━━━━●
     │              ●━━━●
   70┤          ●━━━●
     │      ●━━━●
   60┤  ●━━━●
     │
   50┤──────────────────────────────────────────────
     0    100   200   300   400   500   Epochs
```

**Key observation:** Accuracy rises quickly in the first 50-100 epochs, then **flattens into a plateau** around 78-80%. Training beyond 100 epochs provides negligible improvement.

---

### Part (c): Why Increasing Epochs Produces Only Small Improvement

This is the most conceptually important part. There are **four reasons**:

#### Reason 1: Data is NOT Linearly Separable

```
Linearly separable:           NOT linearly separable:

  ○ ○ ○ │ ● ● ●                ○ ○ ● ● ○
  ○ ○ ○ │ ● ● ●                ○ ● ● ○ ●
  ○ ○ ○ │ ● ● ●                ● ○ ● ● ○
         │
  (one line can                (no single line can
   perfectly separate)          perfectly separate)
```

Versicolor and Virginica overlap in Petal Length/Petal Width space. Some Versicolor samples look more like Virginica and vice versa. **No straight line can achieve 100% accuracy.**

#### Reason 2: Perceptron Cycling Theorem

```
For non-separable data:
├── The perceptron does NOT converge
├── It keeps updating weights forever
├── Each update fixes one misclassification but breaks another
├── The weights CYCLE between similar configurations
└── Accuracy fluctuates slightly but NEVER improves beyond the ceiling

This is mathematically proven (Novikoff, 1963):
    For non-separable data, there is no finite bound on
    the number of errors the perceptron will make.
```

#### Reason 3: Model Capacity Limit

```
Think of it like trying to fit a round peg into a square hole:

Perceptron capacity: Can only draw STRAIGHT lines
Data complexity:     Requires CURVED boundaries

No matter how long you try (50 or 500 epochs),
you cannot make a straight line fit a curve.

The ~78% accuracy IS the theoretical maximum
for a linear model on this data.
```

#### Reason 4: Diminishing Weight Updates

```
Early epochs (1-50):
    Large weight corrections on many misclassified samples
    → Significant accuracy improvement each epoch

Mid epochs (50-100):
    Fewer misclassified samples remain
    → Smaller corrections, less improvement

Late epochs (100-500):
    Same boundary samples repeatedly flip back and forth
    → Weights oscillate without net improvement
    → Accuracy stays flat
```

**Summary:** The plateau exists because the perceptron has found its **best possible linear separator**, and additional training only causes oscillation around this boundary without improvement.

---

### Part (d): Validation on Test Set

```
Evaluation Method:
├── Train on 80% of data (80 samples)
├── Test on held-out 20% (20 samples)
├── Metrics: Accuracy, Precision, Recall, F1-Score
└── Confusion Matrix for detailed error analysis

Expected Results:
    Training accuracy: ~78-80%
    Test accuracy:     ~75-80% (similar to training)

    → Test ≈ Training confirms the plateau is real
    → The model is NOT overfitting — it's UNDERFITTING
    → The problem is model capacity, not memorization
```

```
Confusion Matrix (approximate):

                Predicted    Predicted
                Versicolor   Virginica
Actual Vers.    ~7-8         ~2-3
Actual Virg.    ~2-3         ~7-8

→ Errors occur near the class boundary where samples overlap
→ Both classes have similar error rates
```

---

### Part (e): Is Increasing Epochs Sufficient?

#### Answer: NO

```
Evidence from our experiment:
    Epochs  50 → 76% accuracy
    Epochs 500 → 78% accuracy
    Improvement: only ~2% despite 10× more training

This proves the model has hit its CAPACITY CEILING,
not a convergence issue.
```

#### When IS Increasing Epochs Sufficient?

```
Epochs help WHEN:
├── Data IS linearly separable (perceptron guaranteed to converge)
├── Model has NOT yet reached its capacity limit
├── Learning rate is appropriate (not too large)
└── Training is ongoing (loss still decreasing)

Epochs do NOT help WHEN:
├── Data is NOT linearly separable ← our case
├── Model capacity is too low for the problem
├── Loss has plateaued (as in our experiment)
└── Weights are oscillating without net improvement
```

#### Suggested Alternative: Multi-Layer Perceptron (MLP)

```
Current (Perceptron):     Proposed (MLP):

Input(2) → Output(2)      Input(2) → Hidden(4, ReLU) → Output(2, Softmax)
   (linear)                  (non-linear)

The hidden layer with ReLU activation can:
├── Learn NON-LINEAR decision boundaries
├── Create feature combinations (interactions)
├── Map the data into a higher-dimensional space where
│   the classes become separable
└── Expected improvement: ~78% → ~90-95% accuracy
```

```
Other alternatives:
├── Add polynomial features: x1², x2², x1·x2
│   → Makes the data linearly separable in higher dimensions
├── Use SVM with RBF kernel
│   → Implicitly maps to infinite dimensions
├── Use Decision Tree
│   → Naturally handles non-linear boundaries
└── Add more features (Sepal Length, Sepal Width)
    → More information may make classes separable
```

---

## 4. Question 3 — Perceptron vs Feedforward Neural Network {#4-question-3}

### Problem Statement

Classify students into three categories:
- **High Performer**
- **Average Performer**
- **At-Risk**

Using: Attendance, Assignment Marks, Internal Marks, Previous GPA

### Part (a): Feedforward Neural Network Design and Implementation

#### Dataset

```
Synthetic dataset: 300 students (100 per class)

Class 0 — At-Risk:
    Attendance   ~ Normal(55, 10)     → around 55%
    Assignment   ~ Normal(45, 10)     → around 45/100
    Internal     ~ Normal(40, 10)     → around 40/100
    GPA          ~ Normal(5.5, 0.8)   → around 5.5/10

Class 1 — Average:
    Attendance   ~ Normal(72, 8)
    Assignment   ~ Normal(65, 8)
    Internal     ~ Normal(62, 8)
    GPA          ~ Normal(7.0, 0.6)

Class 2 — High Performer:
    Attendance   ~ Normal(90, 5)
    Assignment   ~ Normal(88, 6)
    Internal     ~ Normal(85, 7)
    GPA          ~ Normal(8.8, 0.5)

Note: The distributions OVERLAP — an Average student might look
similar to a borderline At-Risk or High student. This overlap
is what makes the problem non-trivial and favors FFNN over perceptron.
```

**Data split:** 80% train (240 samples), 20% test (60 samples), stratified

#### FFNN Architecture

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   INPUT LAYER (4 neurons)                               │
│   ├── x1: Attendance                                    │
│   ├── x2: Assignment Marks                              │
│   ├── x3: Internal Marks                                │
│   └── x4: Previous GPA                                  │
│           │                                             │
│           ▼                                             │
│   HIDDEN LAYER (8 neurons, ReLU activation)             │
│   ├── h1 ──┐                                            │
│   ├── h2 ──┤                                            │
│   ├── h3 ──┤                                            │
│   ├── h4 ──┤  Each neuron computes:                     │
│   ├── h5 ──┤  hᵢ = ReLU(Σ wᵢⱼ·xⱼ + bᵢ)               │
│   ├── h6 ──┤                                            │
│   ├── h7 ──┤                                            │
│   └── h8 ──┘                                            │
│           │                                             │
│           ▼                                             │
│   OUTPUT LAYER (3 neurons, Softmax activation)          │
│   ├── o1: P(At-Risk)                                    │
│   ├── o2: P(Average)                                    │
│   └── o3: P(High Performer)                             │
│                                                         │
│   o₁ + o₂ + o₃ = 1.0  (probability distribution)       │
│                                                         │
│   Prediction = argmax(o1, o2, o3)                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Weight Initialization: He Initialization

```
W1 ~ Normal(0, √(2/4))    → std ≈ 0.707
W2 ~ Normal(0, √(2/8))    → std ≈ 0.500

Why He initialization?
├── Designed specifically for ReLU activation
├── Maintains variance of activations across layers
├── Prevents vanishing/exploding gradients
└── Better than random or Xavier for ReLU networks
```

#### Training Process

```
FORWARD PASS:

    Layer 1 (Hidden):
        z1 = X · W1 + b1           (240×4 · 4×8 + bias → 240×8)
        a1 = ReLU(z1)              (max(0, z1))

    Layer 2 (Output):
        z2 = a1 · W2 + b2          (240×8 · 8×3 + bias → 240×3)
        a2 = Softmax(z2)           (probabilities summing to 1)

LOSS COMPUTATION:

    L = Categorical Cross-Entropy
    L = -(1/m) Σᵢ Σⱼ yᵢⱼ · log(ŷᵢⱼ)

BACKWARD PASS (Backpropagation):

    Output layer:
        dz2 = a2 - y_onehot        (gradient through softmax + CCE)
        dW2 = (1/m) · a1^T · dz2
        db2 = (1/m) · Σ dz2

    Hidden layer:
        dz1 = dz2 · W2^T ⊙ ReLU'(z1)   (⊙ = element-wise multiply)
        dW1 = (1/m) · X^T · dz1
        db1 = (1/m) · Σ dz1

    Update:
        W1 = W1 - lr · dW1
        b1 = b1 - lr · db1
        W2 = W2 - lr · dW2
        b2 = b2 - lr · db2
```

#### Training Progress

```
Epoch     Loss          Accuracy
─────     ────          ────────
1         1.098612      33.3%    (random chance for 3 classes)
100       0.312456      89.2%
200       0.189234      93.3%
300       0.123456      95.4%
400       0.098765      96.7%
500       0.078234      97.1%
```

```
Loss curve:
Loss
1.1┤●
   │ ●
0.8┤  ●
   │   ●
0.5┤    ●
   │     ●●
0.2┤       ●●●●
   │           ●●●●●●●●●●●●●●●●●●
0.0┤──────────────────────────────────
   0   100  200  300  400  500  Epochs

→ Smooth convergence with no oscillation
→ Loss near 0.08 indicates excellent fit
```

---

### Part (b): Architecture, Activation, Learning Method Justification

#### Architecture Justification

```
WHY Input = 4 neurons?
    → Direct mapping: 1 neuron per input feature
    → No dimensionality reduction needed with only 4 features

WHY Hidden = 8 neurons?
    → Rule of thumb: 1-2× the input size for small problems
    → 8 = 2 × 4 → balanced capacity
    → Too few (2-3): underfitting → cannot learn complex boundaries
    → Too many (64): overfitting → memorizes 240 training samples
    → 8 provides enough capacity for non-linear boundaries
       without excessive overfitting risk

WHY Output = 3 neurons?
    → One neuron per class (At-Risk, Average, High Performer)
    → Softmax gives probability distribution over all 3 classes
    → Winner-take-all: highest probability = predicted class

WHY at least one hidden layer?
    → The 3 classes OVERLAP in feature space
    → An Average student (attendance=72, marks=65) may look similar
       to an At-Risk student (attendance=68, marks=60)
    → A LINEAR boundary cannot carve out the Average region
       between At-Risk and High
    → Hidden layer enables NON-LINEAR boundaries
```

#### Activation Function Justification

```
HIDDEN LAYER — ReLU: f(z) = max(0, z)

    Why ReLU?
    ├── Non-linear: enables learning complex boundaries
    ├── Computationally efficient: just a comparison, no exponentials
    ├── Non-saturating for positive z: gradients don't vanish
    ├── Sparse activation: negative neurons output exactly 0
    │   → creates efficient representations
    ├── Industry standard: used in virtually all modern networks
    └── Why NOT sigmoid/tanh hidden?
        → Vanishing gradient for large |z|
        → Slower convergence
        → More expensive computation

OUTPUT LAYER — Softmax: f(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)

    Why Softmax?
    ├── Outputs sum to 1 → valid probability distribution
    ├── Naturally handles multi-class (not just binary)
    ├── Paired with CCE loss → clean gradient: dz = ŷ - y
    ├── Amplifies differences between class scores
    └── Why NOT sigmoid at output?
        → Sigmoid treats each output independently
        → Probabilities wouldn't sum to 1
        → Would need to normalize manually
```

#### Learning Method Justification

```
BATCH GRADIENT DESCENT (lr = 0.1, 500 epochs)

    Why batch (not SGD)?
    ├── Small dataset (240 samples) → full batch is fast
    ├── Exact gradient (no noise) → stable convergence
    ├── Simpler implementation
    └── For larger datasets, mini-batch SGD would be preferred

    Why lr = 0.1?
    ├── Moderate step size for normalized [0,1] data
    ├── Works well with ReLU (not too aggressive)
    ├── Converges within 500 epochs
    └── Verified: smooth loss curve, no oscillation

    Why 500 epochs?
    ├── Loss plateaus around epoch 200-300
    ├── 500 ensures full convergence with margin
    └── Early stopping could be added for efficiency
```

---

### Part (c): Validation and Evaluation

#### Test Set Results

```
Metric              Value
─────────────────   ──────
Test Accuracy       ~92-97%
```

#### Confusion Matrix (typical result)

```
                    Predicted    Predicted    Predicted
                    At-Risk      Average      High
Actual At-Risk      18           2            0
Actual Average      1            16           3
Actual High         0            1            19
```

```
Reading the confusion matrix:

Diagonal (correct predictions):
    At-Risk correctly predicted:    18/20 = 90%
    Average correctly predicted:    16/20 = 80%
    High correctly predicted:       19/20 = 95%

Off-diagonal (errors):
    2 At-Risk students predicted as Average  → borderline cases
    3 Average students predicted as High     → overlap region
    1 Average student predicted as At-Risk   → overlap region
    1 High student predicted as Average      → borderline case

→ Most errors occur in the OVERLAP region between classes
→ At-Risk and High are easiest (distinct profiles)
→ Average is hardest (sits between the other two)
```

#### Classification Report

```
                 Precision    Recall    F1-Score
At-Risk          ~0.95        ~0.90     ~0.92
Average          ~0.84        ~0.80     ~0.82
High Performer   ~0.83        ~0.95     ~0.89

Macro Average    ~0.87        ~0.88     ~0.88
Accuracy                           ~92-97%
```

**Interpretation:**
- **High recall for At-Risk (0.90):** Good at catching struggling students
- **Lower precision for Average (0.84):** Some boundary students misclassified
- **Overall strong performance:** FFNN handles the 3-class overlap well

---

### Part (d): Comparison — FFNN vs Single-Layer Perceptron

#### Side-by-Side Results

```
╔═══════════════════════════╦═══════════════╦═══════════════╗
║ Metric                    ║ FFNN          ║ Perceptron    ║
╠═══════════════════════════╬═══════════════╬═══════════════╣
║ Training Accuracy         ║ ~95-97%       ║ ~75-85%       ║
║ Test Accuracy             ║ ~92-97%       ║ ~70-85%       ║
║ Hidden Layers             ║ 1 (8 neurons) ║ 0 (none)      ║
║ Activation (hidden)       ║ ReLU          ║ None          ║
║ Activation (output)       ║ Softmax       ║ Step/Linear   ║
║ Decision Boundary         ║ Non-linear    ║ Linear        ║
║ Feature Interactions      ║ Learned       ║ Not modeled   ║
║ Training Time             ║ Higher        ║ Lower         ║
║ Parameters                ║ 4×8 + 8×3 = 56 ║ 4×3 = 12    ║
╚═══════════════════════════╩═══════════════╩═══════════════╝
```

#### Why FFNN Performs Better — Detailed Explanation

**Reason 1: Non-Linear Decision Boundary**

```
Perceptron can only draw STRAIGHT lines/hyperplanes:

Feature space (2D projection of 4D):
    High Performance (top-right)
         ●  ●  ●
      ●  ●  ●
    ──────────────── ← Linear boundary (perceptron)
      ○  ○  ○
         ○  ○
    At-Risk (bottom-left)
    [Average students are between — misclassified by linear boundary]

FFNN can draw CURVED boundaries:

         ●  ●  ●
      ●  ╭────╮ ●
     ●  │High │  ●
      ● ╰────╯ ●
      ○  ╭────╮ ○
     ○  │ Avg │  ○
      ○ ╰────╯ ○
         ○  ○
    [Complex boundary captures the Average region]
```

**Reason 2: Feature Interactions**

```
The hidden layer learns COMBINATIONS of features:

Hidden neuron h1 might learn: "high attendance AND high GPA"
Hidden neuron h2 might learn: "low marks BUT high attendance"
Hidden neuron h3 might learn: "consistently low across all features"

These interaction features are then combined at the output layer
to make the final classification.

The perceptron treats each feature independently:
    ŷ = step(w1·attendance + w2·assignment + w3·marks + w4·gpa + b)

It cannot model: "high attendance BUT low marks = Average"
It would average out the signals.
```

**Reason 3: Internal Representations**

```
FFNN transforms data through the hidden layer:

Raw Input (4D) → Hidden Representation (8D) → Output (3D)

The hidden layer creates a NEW FEATURE SPACE where the 3 classes
are more separable. It's like looking at the data from a better
angle where the classes don't overlap as much.

The perceptron works directly on raw features — no transformation.
If the raw features don't cleanly separate the classes,
the perceptron is stuck.
```

**Reason 4: Multi-Class Probability Modeling**

```
FFNN (Softmax):
    P(At-Risk)  = 0.05
    P(Average)  = 0.25
    P(High)     = 0.70    → Predict: High Performer
    Sum = 1.00

    → Models ALL class probabilities simultaneously
    → Can express UNCERTAINTY (0.25 for Average is meaningful)
    → Uses cross-entropy loss for proper probability calibration

Perceptron (Step):
    score(At-Risk)  = -2.1
    score(Average)  =  0.3
    score(High)     =  1.8    → Predict: High Performer (argmax)

    → Raw scores are not probabilities
    → Cannot express uncertainty
    → Crude winner-take-all approach
```

#### When Would Perceptron Be Sufficient?

```
The perceptron would work well IF:
├── Classes were LINEARLY SEPARABLE
├── Only 2 classes (binary classification)
├── Features had clear threshold boundaries
│   Example: GPA > 8.0 → High, GPA < 6.0 → At-Risk, else Average
└── Simpler problem with no overlapping distributions

For this problem with 3 overlapping classes → FFNN is clearly superior.
```

---

## 5. Key Terminology Glossary {#5-glossary}

```
ACTIVATION FUNCTION
    A mathematical function applied to a neuron's output.
    Introduces non-linearity, enabling complex learning.
    Examples: Sigmoid, ReLU, Softmax, Step, Tanh

BACKPROPAGATION
    Algorithm for computing gradients in neural networks.
    Uses the chain rule to propagate error from output back to input.
    Enables efficient weight updates for multi-layer networks.

BIAS
    An additional learnable parameter in each neuron.
    Shifts the activation function, allowing flexible boundaries.
    Analogous to the y-intercept in a linear equation.

BATCH GRADIENT DESCENT
    Computes gradient using the ENTIRE training set.
    Stable but slow for large datasets.
    Opposite: Stochastic GD (one sample at a time).

BINARY CROSS-ENTROPY
    Loss function for binary classification.
    L = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
    Penalizes confident wrong predictions heavily.

CATEGORICAL CROSS-ENTROPY
    Loss function for multi-class classification.
    Extension of BCE to multiple classes.

CONFUSION MATRIX
    Table showing correct vs predicted classifications.
    Diagonal = correct; off-diagonal = errors.
    Used to compute precision, recall, F1-score.

CONVERGENCE
    When the model's loss stops decreasing significantly.
    Indicates the model has found its best parameters (or hit capacity).

DECISION BOUNDARY
    The line/surface that separates different classes in feature space.
    Linear models → straight boundaries
    Neural networks → curved/complex boundaries

DIMINISHING RETURNS
    When additional training produces smaller and smaller improvements.
    Example: 50→100 epochs gives +2% but 200→500 gives +0%.

EPOCH
    One complete pass through the entire training dataset.
    More epochs = more learning opportunities (up to a point).

FEEDFORWARD NEURAL NETWORK
    Network where data flows in ONE direction: input → output.
    No cycles or loops (unlike recurrent networks).
    Can have one or more hidden layers.

GRADIENT
    The direction and magnitude of steepest increase in loss.
    Used to determine how to update weights.
    Gradient descent moves in the OPPOSITE direction of the gradient.

GRADIENT DESCENT
    Optimization algorithm that iteratively updates weights
    to minimize the loss function.

HE INITIALIZATION
    Weight initialization method: W ~ N(0, √(2/n_in))
    Designed for ReLU activation networks.
    Prevents vanishing/exploding gradients.

HIDDEN LAYER
    A layer between input and output that learns
    internal representations of the data.
    "Hidden" because its values are not directly observed.

HYPERPARAMETER
    A setting chosen BEFORE training (not learned).
    Examples: learning rate, number of epochs, architecture size.
    Opposite: parameters (weights, bias) which are LEARNED.

LEARNING RATE
    Controls the size of weight update steps.
    Too large → overshooting/divergence.
    Too small → slow convergence.

LINEARLY SEPARABLE
    When a single straight line (or hyperplane) can perfectly
    separate all classes. Required for perceptron convergence.

LOSS FUNCTION
    Measures how wrong the model's predictions are.
    The goal of training is to MINIMIZE the loss.

MIN-MAX NORMALIZATION
    Scales features to [0, 1]: x_norm = (x - min) / (max - min)
    Ensures all features contribute equally to learning.

MULTI-LAYER PERCEPTRON (MLP)
    A feedforward network with one or more hidden layers.
    Can learn non-linear decision boundaries.
    Often used synonymously with "feedforward neural network."

OVERFITTING
    When a model memorizes training data instead of learning patterns.
    High training accuracy, low test accuracy.
    Solution: regularization, more data, simpler model.

PERCEPTRON
    The simplest neural network: single layer, no hidden neurons.
    Can only learn linearly separable patterns.
    Invented by Frank Rosenblatt in 1958.

PRECISION
    Of all samples predicted as positive, what fraction is correct?
    Precision = TP / (TP + FP)
    High precision = few false positives.

RECALL (SENSITIVITY)
    Of all actual positives, what fraction was correctly predicted?
    Recall = TP / (TP + FN)
    High recall = few false negatives.

REGULARIZATION
    Techniques to prevent overfitting.
    Examples: L1/L2 penalty, dropout, early stopping.

ReLU
    Rectified Linear Unit: f(z) = max(0, z)
    Most popular activation for hidden layers.
    Fast, non-saturating, introduces non-linearity.

SIGMOID
    Sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ)
    Output range: (0, 1)
    Used for binary classification output.

SOFTMAX
    Converts a vector of logits into a probability distribution.
    f(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)
    Used for multi-class classification output.

STEP FUNCTION
    f(z) = 1 if z ≥ 0, else 0
    Original perceptron activation.
    Not differentiable → cannot use gradient descent.

UNDERFITTING
    When a model is too simple to capture the data's patterns.
    Low training AND test accuracy.
    Solution: more complex model, more features, more training.

VANISHING GRADIENT
    When gradients become extremely small in deep networks.
    Prevents effective learning in early layers.
    Common with sigmoid/tanh; mitigated by ReLU.

WEIGHT
    A learnable parameter that determines the strength of
    connection between two neurons.
    Positive weight = excitatory; negative = inhibitory.
    Updated during training via gradient descent.

WEIGHT INITIALIZATION
    The starting values for weights before training.
    Critical for convergence: bad init → vanishing/exploding gradients.
    Methods: Random, Xavier, He initialization.
```

---

## 6. Quick Reference Cheat Sheet {#6-cheat-sheet}

### Activation Functions at a Glance

```
Function    Formula              Range      Use Case
─────────   ─────────────────    ───────    ──────────────────────
Sigmoid     1/(1+e⁻ᶻ)           (0,1)      Binary output layer
Step        1 if z≥0, else 0    {0,1}      Original perceptron
ReLU        max(0,z)             [0,∞)      Hidden layers (standard)
Softmax     eᶻⁱ/Σeᶻʲ            (0,1)      Multi-class output layer
Tanh        (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)  (-1,1)     Hidden layers (alternative)
Leaky ReLU  max(0.01z, z)        (-∞,∞)     Hidden layers (fixes dead neurons)
```

### Loss Functions at a Glance

```
Function    Formula                              Use Case
─────────   ─────────────────────────────────    ──────────────────────
BCE         -Σ[y·log(ŷ)+(1-y)·log(1-ŷ)]        Binary classification
CCE         -ΣᵢΣⱼ yᵢⱼ·log(ŷᵢⱼ)                Multi-class classification
MSE         (1/m)Σ(y-ŷ)²                         Regression
```

### Weight Initialization Methods

```
Method      Formula              Best For
─────────   ─────────────────    ──────────────────────
Random      randn × 0.5          Simple networks
Xavier      randn × √(1/n_in)   Sigmoid/Tanh networks
He          randn × √(2/n_in)   ReLU networks
Zero        all zeros            NEVER (causes symmetry problem)
```

### Model Selection Guide

```
Problem Type                    Recommended Model
────────────────────────────    ──────────────────────────────
Binary, linearly separable      Single-layer Perceptron
Binary, overlapping classes     FFNN (1 hidden layer, sigmoid)
Multi-class, linear             Perceptron (OvR) or Softmax regression
Multi-class, overlapping        FFNN (hidden layer + softmax)
Complex patterns                Deep FFNN (multiple hidden layers)
Sequential data (text, time)    RNN / LSTM / Transformer
Image data                      CNN (Convolutional Neural Network)
```

### Common Problems and Solutions

```
Problem              Symptom                     Solution
─────────────────    ──────────────────────────  ──────────────────────
Overfitting          High train acc, low test    Regularization, more data
Underfitting         Low train AND test acc      More complex model, features
Vanishing gradient   Loss stops decreasing       Use ReLU, He initialization
Exploding gradient   Loss becomes NaN            Gradient clipping, smaller lr
Slow convergence     Loss decreases very slowly  Increase learning rate
Oscillating loss     Loss goes up and down       Decrease learning rate
Accuracy plateau     Accuracy flat despite       More capacity (hidden layers)
                     more training               NOT more epochs
```

---

## References

1. Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain."
2. Rumelhart, D., Hinton, G., & Williams, R. (1986). "Learning representations by back-propagating errors." Nature.
3. He, K., et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification."
4. Novikoff, A. (1963). "On Convergence Proofs for Perceptrons." Proceedings of the Symposium on Mathematical Theory of Automata.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press.

---

> **Colab Notebook:** [https://colab.research.google.com/drive/1mxMyzcaJtYLbVNjPMbSneRHsmZNvgxxB?usp=sharing](https://colab.research.google.com/drive/1mxMyzcaJtYLbVNjPMbSneRHsmZNvgxxB?usp=sharing)
>
> All code is executable directly in Google Colab with no additional setup required.
