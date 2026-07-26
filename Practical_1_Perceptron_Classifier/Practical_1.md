# Perceptron Classifier — Student Pass Prediction

A from-scratch Perceptron classifier built using **NumPy** that predicts whether a student passes a course based on two features:

- Study Hours
- Class Attendance

The project demonstrates how a single-layer perceptron learns a linear decision boundary, along with its limitations when faced with overlapping or imbalanced datasets.

---

# How It Works

The perceptron learns a straight-line boundary in a two-dimensional feature space that separates **PASS** from **FAIL**.

For every training sample:

1. Compute the weighted sum of the inputs.
2. Apply the step activation function.
3. Compare the prediction with the actual label.
4. Update the weights only if the prediction is incorrect.

Mathematically:

```text
prediction = step(w1 × study_hours + w2 × attendance + bias)

if prediction != target:
    w = w + learning_rate × error × input
    bias = bias + learning_rate × error
```

Over multiple epochs, the decision boundary gradually shifts until it separates the two classes as well as possible.

---

# Project Workflow

| Step | Purpose |
|------|---------|
| Data generation | Create synthetic student records with PASS/FAIL labels |
| Normalization | Standardize features (mean = 0, std = 1) so both contribute equally |
| Train/Test Split | Perform an 80/20 split before normalization to avoid data leakage |
| Training | Shuffle samples every epoch and update weights after mistakes |
| Evaluation | Compute Accuracy, Precision, Recall, and Confusion Matrix |
| Prediction | Predict outcomes for previously unseen students |

---

# Baseline Results

Using a balanced dataset:

- 50 FAIL students
- 50 PASS students

with clearly separated feature distributions, the perceptron converges quickly.

Example output:

```text
Learned weights: [0.277, 0.296]
Learned bias:    0.200
```

Both learned weights are positive, indicating that:

- More study hours increase the chance of passing.
- Better attendance increases the chance of passing.

The model reaches **100% test accuracy** after approximately **12 epochs** because the data is linearly separable.

---

# Experiments: Making the Problem Harder

To explore the limitations of the perceptron, two changes were introduced simultaneously.

## 1. Class Imbalance

The dataset was modified from

- 50 FAIL / 50 PASS

to

- 80 FAIL / 20 PASS

```python
fail_study = np.random.normal(loc=3, scale=1.5, size=80)
pass_study = np.random.normal(loc=8, scale=2, size=20)
```

This creates a strong bias toward the FAIL class.

---

## 2. Overlapping Distributions

Originally:

- FAIL attendance ≈ 55%
- PASS attendance ≈ 80%

The PASS attendance distribution was moved closer to FAIL students.

```python
pass_attend = np.random.normal(loc=68, scale=10, size=50)
```

The two classes now overlap considerably, making them much harder to separate with a single straight line.

---

# Training Results

Example output:

```text
Epoch  10 | Errors: 16 | Accuracy: 80.0%
Epoch  50 | Errors: 22 | Accuracy: 72.5%
Epoch 100 | Errors: 17 | Accuracy: 78.8%

WARNING: Attendance weight is negative!

Test Accuracy: 70.0%
Precision:     20.0%
Recall:        33.3%
```

Unlike the balanced dataset, the perceptron never fully converges.

---

# Problems Observed

| Observation | Explanation |
|-------------|-------------|
| Training never converges | The classes overlap, so no single straight line can perfectly separate them. |
| Negative attendance weight | The heavy class imbalance causes the model to learn an unrealistic decision boundary that predicts FAIL most of the time. |
| Precision = 20% | Most predicted PASS students are actually FAIL students. The PASS predictions become unreliable. |

---

# Hidden Data Bug

An additional issue existed in the synthetic dataset.

The class imbalance was only applied to **study hours**, while attendance still contained **50 FAIL** and **50 PASS** samples.

Study Hours:

```python
80 FAIL
20 PASS
```

Attendance:

```python
50 FAIL
50 PASS
```

Labels:

```python
50 FAIL
50 PASS
```

When these arrays are combined column-wise, rows **50–79** become inconsistent.

Those rows contain:

- FAIL-level study hours
- PASS labels

This introduces incorrect training examples.

Conceptually:

```text
Study Hours : [80 FAIL | 20 PASS]
Attendance  : [50 FAIL | 50 PASS]
Labels      : [50 FAIL | 50 PASS]

Rows 50–79:
    FAIL study hours
    PASS labels
```

The perceptron attempts to learn from contradictory samples, introducing additional noise beyond the already overlapping distributions.

---

# Correct Fix

All feature arrays should contain the same number of samples.

```python
fail_study  = np.random.normal(loc=3, scale=1.5, size=80)
fail_attend = np.random.normal(loc=55, scale=12, size=80)

pass_study  = np.random.normal(loc=8, scale=2, size=20)
pass_attend = np.random.normal(loc=68, scale=10, size=20)
```

Now every feature corresponds to the correct label.

---

# Lessons Learned

This experiment illustrates several important machine learning concepts.

## 1. Accuracy Can Be Misleading

An accuracy of 70–80% appears reasonable.

However, if 80% of all students belong to the FAIL class, a model that predicts FAIL for every student already achieves approximately 80% accuracy.

This is why additional metrics such as Precision and Recall are essential.

---

## 2. Inspect the Learned Weights

The learned parameters should make logical sense.

A negative attendance weight implies that attending more classes decreases the chance of passing, which contradicts the problem itself.

Weight inspection often reveals issues that evaluation metrics alone cannot.

---

## 3. Perceptrons Cannot Handle Overlapping Classes Well

A perceptron learns only a **linear decision boundary**.

When the classes overlap significantly, no straight line can perfectly separate them.

In such cases, more expressive models are required, such as:

- Multi-Layer Perceptrons (MLPs)
- Neural Networks
- Support Vector Machines with non-linear kernels

---

## 4. Synthetic Data Must Be Consistent

Every feature array and the label array must contain the same number of rows.

Even small indexing mismatches silently introduce incorrect training examples that can severely degrade model performance.

---

# Dependencies

Only **NumPy** is required.

No additional machine learning libraries are used.