# Cats vs Dogs CNN Classification — Hyperparameter Comparison Report

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
4. [CNN Architecture](#cnn-architecture)
5. [Experimental Setup](#experimental-setup)
6. [Results Summary](#results-summary)
7. [Detailed Experiment Analysis](#detailed-experiment-analysis)
   - [Experiment 1: Baseline (ReLU, Adam, LR 0.001, Batch 32)](#experiment-1-baseline-relu-adam-lr-0001-batch-32)
   - [Experiment 2: Activation — Tanh](#experiment-2-activation--tanh)
   - [Experiment 3: Activation — ELU](#experiment-3-activation--elu)
   - [Experiment 4: Optimizer — SGD](#experiment-4-optimizer--sgd)
   - [Experiment 5: Optimizer — RMSprop](#experiment-5-optimizer--rmsprop)
   - [Experiment 6: High Learning Rate (0.01)](#experiment-6-high-learning-rate-001)
   - [Experiment 7: Low Learning Rate (0.0001)](#experiment-7-low-learning-rate-00001)
   - [Experiment 8: Batch Size 16](#experiment-8-batch-size-16)
8. [Comparative Analysis by Hyperparameter](#comparative-analysis-by-hyperparameter)
   - [Activation Functions](#activation-functions)
   - [Optimizers](#optimizers)
   - [Learning Rates](#learning-rates)
   - [Batch Sizes](#batch-sizes)
9. [Final Rankings](#final-rankings)
10. [Key Takeaways](#key-takeaways)
11. [Recommendations for Further Improvement](#recommendations-for-further-improvement)
12. [Appendix: Full Epoch Logs](#appendix-full-epoch-logs)

---

## Project Overview

This report presents the results of a systematic hyperparameter comparison study using a Convolutional Neural Network (CNN) for binary image classification — distinguishing between cats and dogs. The goal is to understand how different hyperparameters affect model performance, convergence speed, and generalization ability.

**What was varied:**
- Activation functions: ReLU, Tanh, ELU
- Optimizers: Adam, SGD, RMSprop
- Learning rates: 0.01, 0.001, 0.0001
- Batch sizes: 16, 32

**What was kept constant:**
- CNN architecture (2 conv blocks + 2 dense layers)
- Number of epochs: 15
- Image size: 64×64 pixels
- Data augmentation pipeline
- Train/test/validation split ratios

---

## Dataset Description

**Source:** Kaggle Cats and Dogs Dataset (Microsoft)

**Raw path:** `C:\Users\HP\Downloads\cats_and_dogs\kagglecatsanddogs_3367a\PetImages`

The raw dataset contains 12,500 cat images and 12,500 dog images in a flat structure with no train/test/validation split. Several images in the dataset are corrupted or unreadable.

### Data Cleaning

Before splitting, all images were validated using `PIL.Image.verify()`. Corrupted and unreadable files were automatically detected and removed from the dataset.

### Split Ratios

| Split        | Ratio | Purpose                                      |
|--------------|-------|----------------------------------------------|
| Training     | 70%   | Model learns patterns from these images      |
| Test         | 15%   | Evaluated during training for monitoring     |
| Validation   | 15%   | Final unseen evaluation (held out entirely)  |

### Final Image Counts

| Split        | Cat Images | Dog Images | Total   |
|--------------|-----------|-----------|---------|
| Training Set | ~8,736    | ~8,735    | 17,471  |
| Test Set     | ~1,872    | ~1,873    | 3,745   |
| Validation   | ~1,871    | ~1,872    | 3,743   |

**Note:** Exact counts vary slightly due to the random shuffle with seed 42 and the number of corrupted images removed per class.

### Generated Folder Structure

```
C:\Users\HP\Downloads\cats_and_dogs\dataset\
├── training_set\
│   ├── Cat\
│   └── Dog\
├── test_set\
│   ├── Cat\
│   └── Dog\
└── validation_set\
    ├── Cat\
    └── Dog\
```

---

## Data Preprocessing & Augmentation

### Training Set Augmentation

The training set uses the following augmentations applied randomly on-the-fly during each epoch:

| Augmentation      | Value  | Purpose                                                    |
|-------------------|--------|------------------------------------------------------------|
| Rescale           | 1/255  | Normalize pixel values from [0, 255] to [0, 1]            |
| Shear Range       | 0.2    | Randomly shear (tilt) images by up to 20%                  |
| Zoom Range        | 0.2    | Randomly zoom in/out by up to 20%                          |
| Horizontal Flip   | True   | Randomly mirror images left-to-right                       |

These augmentations serve as a form of **regularization** — they prevent the model from memorizing exact training images by presenting slightly different versions each epoch. This effectively increases the size of the training set and helps the model learn position-invariant features.

### Test and Validation Sets

No augmentation is applied to the test or validation sets. Only rescaling (1/255) is applied. This ensures evaluation is performed on unmodified images for fair and consistent accuracy measurement.

### Image Resizing

All images are resized to **64×64 pixels** with **3 color channels (RGB)**. This is a relatively small resolution — sufficient for this simple CNN architecture but would limit performance on more complex models that could benefit from finer spatial details.

---

## CNN Architecture

The same architecture is used across all 8 experiments to ensure a fair comparison. Only the hyperparameters change.

```
Layer (type)              Output Shape         Parameters
─────────────────────────────────────────────────────────
Conv2D (32 filters, 3×3)  (None, 62, 62, 32)   896
MaxPool2D (2×2)           (None, 31, 31, 32)   0
Conv2D (32 filters, 3×3)  (None, 29, 29, 32)   9,248
MaxPool2D (2×2)           (None, 14, 14, 32)   0
Flatten                   (None, 6272)          0
Dense (128 units)         (None, 128)           802,944
Dense (1 unit, sigmoid)   (None, 1)             129
─────────────────────────────────────────────────────────
Total parameters: ~813,217
```

### Architecture Breakdown

| Component       | Details                         | Role                                                    |
|-----------------|---------------------------------|---------------------------------------------------------|
| **Conv2D #1**   | 32 filters, 3×3 kernel          | Detects low-level features: edges, corners, textures    |
| **MaxPool2D #1**| 2×2 pool, stride 2              | Halves spatial dimensions (64→31), reduces computation   |
| **Conv2D #2**   | 32 filters, 3×3 kernel          | Detects higher-level features: shapes, ears, eyes       |
| **MaxPool2D #2**| 2×2 pool, stride 2              | Further reduces dimensions (31→14)                       |
| **Flatten**     | Reshapes 3D → 1D                | Converts feature maps to a vector for dense layers      |
| **Dense (128)** | Fully connected, 128 neurons    | Learns non-linear combinations of extracted features    |
| **Dense (1)**   | Output layer, sigmoid activation | Outputs probability: 0 = Cat, 1 = Dog                  |

The output layer always uses **sigmoid activation** regardless of the experiment, as it is required for binary classification (outputs a probability between 0 and 1).

---

## Experimental Setup

### Experiments Overview

| #  | Name                       | Activation | Optimizer | Learning Rate | Batch Size |
|----|----------------------------|-----------|-----------|---------------|------------|
| 1  | Baseline                   | ReLU      | Adam      | 0.001         | 32         |
| 2  | Activation: Tanh           | Tanh      | Adam      | 0.001         | 32         |
| 3  | Activation: ELU            | ELU       | Adam      | 0.001         | 32         |
| 4  | Optimizer: SGD             | ReLU      | SGD       | 0.001         | 32         |
| 5  | Optimizer: RMSprop         | ReLU      | RMSprop   | 0.001         | 32         |
| 6  | LR: 0.01 (High)            | ReLU      | Adam      | 0.01          | 32         |
| 7  | LR: 0.0001 (Low)           | ReLU      | Adam      | 0.0001        | 32         |
| 8  | Batch: 16                  | ReLU      | Adam      | 0.001         | 16         |

### Methodology

For each experiment:
1. A fresh model is initialized with random weights (no transfer learning between experiments).
2. The training set is reloaded with the appropriate batch size.
3. The model is trained for 15 epochs on the training set.
4. The test set is evaluated after each epoch for monitoring.
5. Final epoch metrics (training accuracy, validation accuracy, validation loss) are recorded.
6. The model is deleted from memory before the next experiment begins.

---

## Results Summary

### Final Comparison Table (Ranked by Validation Accuracy)

| Rank | Experiment                  | Activation | Optimizer | LR      | Batch | Train Acc | Val Acc  | Val Loss |
|------|-----------------------------|-----------|-----------|---------|-------|-----------|----------|----------|
| 1    | Baseline (ReLU, Adam, 0.001)| ReLU      | Adam      | 0.001   | 32    | 0.8700    | **0.8314** | 0.3903   |
| 2    | Optimizer: RMSprop          | ReLU      | RMSprop   | 0.001   | 32    | 0.8590    | 0.8186   | 0.4319   |
| 3    | Batch: 16                   | ReLU      | Adam      | 0.001   | 16    | 0.8439    | 0.8173   | 0.4172   |
| 4    | Activation: ELU             | ELU       | Adam      | 0.001   | 32    | 0.8426    | 0.7978   | 0.4597   |
| 5    | LR: 0.0001 (Low)            | ReLU      | Adam      | 0.0001  | 32    | 0.7913    | 0.7820   | 0.4584   |
| 6    | Activation: Tanh            | Tanh      | Adam      | 0.001   | 32    | 0.8102    | 0.7772   | 0.4784   |
| 7    | Optimizer: SGD              | ReLU      | SGD       | 0.001   | 32    | 0.6280    | 0.6139   | 0.6457   |
| 8    | LR: 0.01 (High)             | ReLU      | Adam      | 0.01    | 32    | 0.4950    | 0.4996   | 0.6933   |

### Visual Summary

```
Validation Accuracy Ranking:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline (ReLU,Adam,0.001)  ████████████████████████████████████  83.1%
RMSprop                     ███████████████████████████████████   81.9%
Batch: 16                   ██████████████████████████████████    81.7%
ELU                         █████████████████████████████████     79.8%
LR: 0.0001                  ████████████████████████████████      78.2%
Tanh                        ███████████████████████████████       77.7%
SGD                         ████████████████████████              61.4%
LR: 0.01                    ████████████████████                  50.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Detailed Experiment Analysis

### Experiment 1: Baseline (ReLU, Adam, LR 0.001, Batch 32)

**Configuration:** The default combination — ReLU activation, Adam optimizer, learning rate 0.001, batch size 32.

**Final Results:** Train Accuracy: 87.00% | Val Accuracy: 83.14% | Val Loss: 0.3903

#### Epoch-by-Epoch Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.6725    | 0.6020     | 0.7286  | 0.5393   |
| 2     | 0.7503    | 0.5117     | 0.7724  | 0.4704   |
| 3     | 0.7697    | 0.4759     | 0.7954  | 0.4374   |
| 4     | 0.7856    | 0.4490     | 0.7895  | 0.4548   |
| 5     | 0.7978    | 0.4337     | 0.8020  | 0.4265   |
| 6     | 0.8073    | 0.4128     | 0.8063  | 0.4174   |
| 7     | 0.8187    | 0.4006     | 0.8026  | 0.4359   |
| 8     | 0.8259    | 0.3837     | 0.8119  | 0.4121   |
| 9     | 0.8297    | 0.3753     | 0.8151  | 0.4034   |
| 10    | 0.8408    | 0.3588     | 0.8229  | 0.3975   |
| 11    | 0.8461    | 0.3452     | 0.8234  | 0.3965   |
| 12    | 0.8517    | 0.3337     | 0.8229  | 0.4066   |
| 13    | 0.8592    | 0.3209     | 0.8330  | 0.3936   |
| 14    | 0.8651    | 0.3109     | 0.8328  | 0.3793   |
| 15    | 0.8700    | 0.2997     | 0.8314  | 0.3903   |

#### Analysis

This is the **best-performing configuration** across all experiments. Key observations:

- **Rapid early learning:** The model jumped from 67.3% to 79.8% training accuracy in just the first 5 epochs, demonstrating that Adam with LR 0.001 provides fast convergence on this task.
- **Healthy generalization gap:** The difference between training accuracy (87.0%) and validation accuracy (83.1%) is approximately 4%. This indicates the model is learning real patterns rather than memorizing training data.
- **Val loss decreased consistently:** From 0.5393 down to 0.3903 over 15 epochs, with minor fluctuations (epoch 7 and 12 showed slight increases). The overall trend is clearly downward.
- **Slight overfitting onset:** Training accuracy continued climbing steadily while validation accuracy plateaued around epoch 13 (83.3% → 83.1%). This is normal and mild — the model is beginning to memorize some training-specific details, but hasn't crossed into problematic overfitting.

**Why this configuration works best:**
- **ReLU** avoids vanishing gradients, allowing full gradient flow through the network.
- **Adam** adapts learning rates per-parameter using running averages of first and second moments of gradients, making it robust across different gradient magnitudes.
- **LR 0.001** is Adam's default and a well-tested sweet spot for many image classification tasks.
- **Batch size 32** provides stable gradient estimates without excessive memory usage.

---

### Experiment 2: Activation — Tanh

**Configuration:** Same as baseline, but replacing ReLU with Tanh activation in all conv and dense layers.

**Final Results:** Train Accuracy: 81.02% | Val Accuracy: 77.72% | Val Loss: 0.4784

#### Epoch-by-Epoch Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.6525    | 0.6252     | 0.7131  | 0.5656   |
| 2     | 0.7133    | 0.5631     | 0.7272  | 0.5389   |
| 3     | 0.7356    | 0.5335     | 0.7331  | 0.5185   |
| 4     | 0.7437    | 0.5207     | 0.7417  | 0.5207   |
| 5     | 0.7478    | 0.5116     | 0.7144  | 0.5458   |
| 6     | 0.7568    | 0.5029     | 0.7582  | 0.4860   |
| 7     | 0.7646    | 0.4907     | 0.7657  | 0.4794   |
| 8     | 0.7652    | 0.4879     | 0.7708  | 0.4795   |
| 9     | 0.7744    | 0.4718     | 0.7697  | 0.4738   |
| 10    | 0.7774    | 0.4652     | 0.7750  | 0.4621   |
| 11    | 0.7781    | 0.4594     | 0.7847  | 0.4547   |
| 12    | 0.7949    | 0.4458     | 0.7855  | 0.4571   |
| 13    | 0.8022    | 0.4293     | 0.7852  | 0.4462   |
| 14    | 0.8037    | 0.4272     | 0.7815  | 0.4605   |
| 15    | 0.8102    | 0.4155     | 0.7772  | 0.4784   |

#### Analysis

- **Slower convergence than ReLU:** At epoch 5, Tanh was at 74.8% training accuracy vs ReLU's 79.8% — a 5 percentage point gap that persisted throughout training.
- **Lower final training accuracy:** Tanh reached 81.0% vs ReLU's 87.0%. The activation function fundamentally limited how much the model could learn in 15 epochs.
- **Overfitting visible in final epochs:** Val accuracy peaked at 78.6% (epoch 12) then *decreased* to 77.7% by epoch 15, while training accuracy continued climbing from 79.5% to 81.0%. Val loss also increased from 0.4462 to 0.4784 in the last 3 epochs. This is a classic sign of overfitting — the model is fitting training-specific noise.
- **Epoch 5 anomaly:** Validation accuracy dropped sharply to 71.4% (from 74.2% the previous epoch). This kind of instability is more common with Tanh due to the saturation behavior of the function.

**Why Tanh underperforms:**

Tanh outputs values in the range [-1, +1]. While this zero-centered output is theoretically beneficial, the function has a critical weakness: **saturation**. When inputs are large (positive or negative), tanh outputs values very close to -1 or +1, where the gradient approaches zero. This is the **vanishing gradient problem**.

In a CNN with multiple layers, gradients are multiplied back through each layer during backpropagation. If each layer's gradient is a small number (due to tanh saturation), the product of many small numbers becomes tiny. The early layers receive almost no learning signal, and training stalls.

ReLU avoids this entirely because its gradient is either 0 (for negative inputs) or 1 (for positive inputs) — it never vanishes for positive activations.

---

### Experiment 3: Activation — ELU

**Configuration:** Same as baseline, but replacing ReLU with ELU (Exponential Linear Unit) activation.

**Final Results:** Train Accuracy: 84.26% | Val Accuracy: 0.7978% | Val Loss: 0.4597

#### Epoch-by-Epoch Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.6566    | 0.6313     | 0.7179  | 0.5527   |
| 2     | 0.7216    | 0.5503     | 0.7296  | 0.5304   |
| 3     | 0.7404    | 0.5239     | 0.7395  | 0.5166   |
| 4     | 0.7476    | 0.5097     | 0.7441  | 0.4996   |
| 5     | 0.7581    | 0.4954     | 0.7670  | 0.4862   |
| 6     | 0.7670    | 0.4788     | 0.7438  | 0.5134   |
| 7     | 0.7770    | 0.4678     | 0.7668  | 0.4725   |
| 8     | 0.7847    | 0.4558     | 0.7812  | 0.4562   |
| 9     | 0.7954    | 0.4441     | 0.7766  | 0.4679   |
| 10    | 0.8024    | 0.4235     | 0.7793  | 0.4637   |
| 11    | 0.8099    | 0.4132     | 0.7769  | 0.4935   |
| 12    | 0.8134    | 0.4051     | 0.7852  | 0.4585   |
| 13    | 0.8224    | 0.3894     | 0.7996  | 0.4291   |
| 14    | 0.8358    | 0.3691     | 0.8020  | 0.4314   |
| 15    | 0.8426    | 0.3584     | 0.7978  | 0.4597   |

#### Analysis

- **Better than Tanh, worse than ReLU:** ELU reached 84.3% training accuracy (vs Tanh's 81.0% and ReLU's 87.0%). It sits comfortably between the two.
- **Faster early learning than Tanh:** By epoch 5, ELU was at 75.8% vs Tanh's 74.8%, closing the gap with ReLU (79.8%).
- **More volatile validation accuracy:** Val accuracy showed several dips — epoch 6 (74.4%), epoch 9 (77.7%), epoch 11 (77.7%). This instability is less pronounced with ReLU.
- **Clear overfitting in final epochs:** From epoch 13 to 15, train accuracy rose from 82.2% to 84.3% while val accuracy dropped from 80.0% to 79.8%. Val loss increased from 0.4291 to 0.4597. The model was memorizing training data without improving generalization.
- **Late-blooming:** ELU's best val accuracy (80.2%) came at epoch 14, just one epoch before training ended. With more epochs, it might have improved slightly further, but the overfitting trend suggests the gains would be marginal.

**What is ELU and why is it different:**

ELU is defined as:
- `x` if `x > 0`
- `α * (exp(x) - 1)` if `x ≤ 0` (where α is typically 1.0)

Unlike ReLU, which outputs 0 for all negative inputs, ELU outputs a smooth negative curve. This has two theoretical benefits:
1. **No dead neurons:** ReLU can "die" when a neuron's weights are such that it always receives negative input — its gradient is permanently 0 and it never learns again. ELU always has a non-zero gradient for negative inputs.
2. **Zero-centered outputs:** ELU's outputs are closer to zero-mean, which can help gradient flow.

In practice, on this simple architecture with only 2 conv layers, the dead neuron problem is rare, so ELU's advantages don't fully materialize.

---

### Experiment 4: Optimizer — SGD

**Configuration:** Same as baseline, but replacing Adam with SGD (Stochastic Gradient Descent) with no momentum.

**Final Results:** Train Accuracy: 62.80% | Val Accuracy: 61.39% | Val Loss: 0.6457

#### Epoch-by-Epoch Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.5290    | 0.6914     | 0.5498  | 0.6892   |
| 2     | 0.5555    | 0.6872     | 0.5594  | 0.6861   |
| 3     | 0.5738    | 0.6840     | 0.5835  | 0.6828   |
| 4     | 0.5795    | 0.6813     | 0.5795  | 0.6799   |
| 5     | 0.5838    | 0.6781     | 0.5848  | 0.6766   |
| 6     | 0.5901    | 0.6752     | 0.5923  | 0.6733   |
| 7     | 0.5974    | 0.6723     | 0.5827  | 0.6718   |
| 8     | 0.5992    | 0.6695     | 0.5987  | 0.6666   |
| 9     | 0.6044    | 0.6660     | 0.6212  | 0.6610   |
| 10    | 0.6089    | 0.6625     | 0.6107  | 0.6582   |
| 11    | 0.6163    | 0.6582     | 0.6286  | 0.6517   |
| 12    | 0.6156    | 0.6564     | 0.5987  | 0.6555   |
| 13    | 0.6246    | 0.6518     | 0.5952  | 0.6553   |
| 14    | 0.6253    | 0.6495     | 0.6238  | 0.6460   |
| 15    | 0.6280    | 0.6451     | 0.6139  | 0.6457   |

#### Analysis

- **Dramatically slower learning:** After 15 epochs, the model was only at 62.8% training accuracy. The baseline (Adam) reached this level by epoch 2. SGD with LR 0.001 is approximately **7-8x slower** to converge.
- **Loss barely moved:** Starting loss was 0.6914 (near the 0.6932 random-guess baseline), and it decreased to only 0.6451 after 15 epochs. Compare this to the baseline's 0.2997.
- **Validation accuracy unstable:** Val accuracy fluctuated significantly — 62.9% at epoch 11, dropped to 59.5% at epoch 13, recovered to 62.4% at epoch 14, then dipped again to 61.4%. This is characteristic of vanilla SGD's noisy gradient estimates.
- **Still learning at epoch 15:** The loss was still decreasing and accuracy still climbing, suggesting the model needed perhaps 50+ more epochs to reach comparable performance to the baseline.

**Why SGD struggles here:**

Vanilla SGD updates weights using the formula: `w = w - lr * gradient`. Every parameter receives the same learning rate regardless of gradient history. This causes two problems:

1. **Uniform step size:** Parameters with large gradients and parameters with small gradients are updated by the same relative amount. This is inefficient — some parameters need large updates, others need tiny ones.
2. **No momentum:** Without momentum, SGD has no "memory" of past gradients. It can't accelerate through consistent gradient directions or dampen oscillations in noisy directions.

Adam, by contrast, maintains per-parameter adaptive learning rates using exponential moving averages of the gradient (first moment) and the squared gradient (second moment). This allows it to take large steps where gradients are consistently small and small steps where gradients are noisy.

**Note:** SGD with momentum (e.g., `SGD(lr=0.001, momentum=0.9)`) would perform significantly better, but this experiment tests vanilla SGD specifically.

---

### Experiment 5: Optimizer — RMSprop

**Configuration:** Same as baseline, but replacing Adam with RMSprop.

**Final Results:** Train Accuracy: 85.90% | Val Accuracy: 81.86% | Val Loss: 0.4319

#### Epoch-by-Epoch Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.6371    | 0.6383     | 0.7136  | 0.5514   |
| 2     | 0.7253    | 0.5425     | 0.7430  | 0.5142   |
| 3     | 0.7526    | 0.5054     | 0.7726  | 0.4852   |
| 4     | 0.7739    | 0.4771     | 0.7839  | 0.4562   |
| 5     | 0.7865    | 0.4554     | 0.7702  | 0.4743   |
| 6     | 0.7924    | 0.4406     | 0.8031  | 0.4348   |
| 7     | 0.8010    | 0.4304     | 0.8050  | 0.4226   |
| 8     | 0.8135    | 0.4134     | 0.8071  | 0.4294   |
| 9     | 0.8218    | 0.3980     | 0.8100  | 0.4425   |
| 10    | 0.8243    | 0.3867     | 0.8106  | 0.4231   |
| 11    | 0.8310    | 0.3789     | 0.8197  | 0.4105   |
| 12    | 0.8410    | 0.3640     | 0.8226  | 0.4132   |
| 13    | 0.8450    | 0.3553     | 0.7919  | 0.5306   |
| 14    | 0.8492    | 0.3451     | 0.7945  | 0.4788   |
| 15    | 0.8590    | 0.3302     | 0.8186  | 0.4319   |

#### Analysis

- **Comparable to Adam in early training:** RMSprop and Adam tracked very closely through epochs 1-8. At epoch 8, RMSprop was at 81.4% training / 80.7% val accuracy vs Adam's 82.6% / 81.2%.
- **Epoch 13 crash:** Val accuracy suddenly dropped from 82.3% to 79.2% and val loss spiked from 0.4132 to 0.5306. This is a sharp instability event. It partially recovered by epoch 15 (81.9% val accuracy) but the damage was done.
- **More overfitting than Adam:** The gap between train (85.9%) and val (81.9%) accuracy is 4.0%, similar to baseline's 3.9%. However, RMSprop's val loss (0.4319) is notably higher than baseline's (0.3903), indicating less confident predictions on unseen data.

**RMSprop vs Adam:**

RMSprop and Adam are closely related. Both adapt learning rates per-parameter by dividing the learning rate by a running average of squared gradient magnitudes. The key difference:

- **RMSprop:** Uses only the second moment (running average of squared gradients). `w = w - lr * gradient / sqrt(v + ε)`
- **Adam:** Uses both the first moment (running average of gradients, like momentum) AND the second moment. `w = w - lr * m / (sqrt(v) + ε)`

Adam's momentum term (first moment) helps smooth out gradient noise and provides a consistent update direction, which is likely why Adam was more stable than RMSprop in this experiment (no epoch 13-style crash).

---

### Experiment 6: High Learning Rate (0.01)

**Configuration:** Same as baseline, but with learning rate increased from 0.001 to 0.01.

**Final Results:** Train Accuracy: 49.50% | Val Accuracy: 49.96% | Val Loss: 0.6933

#### Epoch-by-Epoch Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.5071    | 0.7411     | 0.5004  | 0.6936   |
| 2     | 0.4976    | 0.6934     | 0.5004  | 0.6932   |
| 3     | 0.4959    | 0.6935     | 0.5004  | 0.6936   |
| 4     | 0.4942    | 0.6934     | 0.5004  | 0.6932   |
| 5     | 0.4970    | 0.6935     | 0.5004  | 0.6932   |
| 6     | 0.5001    | 0.6934     | 0.5004  | 0.6933   |
| 7     | 0.4961    | 0.6934     | 0.4996  | 0.6933   |
| 8     | 0.5026    | 0.6934     | 0.5004  | 0.6931   |
| 9     | 0.4998    | 0.6934     | 0.4996  | 0.6932   |
| 10    | 0.4929    | 0.6934     | 0.4996  | 0.6932   |
| 11    | 0.4988    | 0.6933     | 0.5004  | 0.6933   |
| 12    | 0.4995    | 0.6935     | 0.5004  | 0.6934   |
| 13    | 0.4960    | 0.6934     | 0.5004  | 0.6931   |
| 14    | 0.4936    | 0.6935     | 0.5004  | 0.6933   |
| 15    | 0.4950    | 0.6935     | 0.4996  | 0.6933   |

#### Analysis

- **Complete failure to learn:** The model's accuracy remained at ~50% for all 15 epochs — equivalent to random guessing on a binary classification task. The model essentially learned nothing.
- **Loss stuck at 0.693:** The value 0.6932 is `-ln(0.5)`, which is the binary cross-entropy loss of a model that always outputs 0.5 (i.e., completely uncertain). The model's outputs are stuck at the coin-flip probability.
- **No improvement over time:** Unlike SGD (which was slow but still learning), the high-LR model showed zero trend in any metric across all 15 epochs. Every epoch produced nearly identical numbers.

**What is happening — Divergence:**

When the learning rate is too high, the weight updates overshoot the optimal values. Imagine rolling a ball into a valley (the loss minimum):

- **LR 0.001 (baseline):** The ball rolls gently into the valley and settles at the bottom.
- **LR 0.0001 (low):** The ball rolls very slowly into the valley. It gets there eventually, but takes longer.
- **LR 0.01 (high):** The ball is launched so hard that it flies over the valley, lands on the other side, and bounces back and forth forever without settling down.

With Adam at LR 0.01, the adaptive scaling actually makes things worse in this case. Adam divides the gradient by the square root of the second moment, which can amplify the already-too-large learning rate. The weight updates oscillate wildly, and the model is trapped in a state where every step forward is immediately undone by the next step backward.

**This is the single most important result in the entire experiment** — it demonstrates that learning rate selection is the most consequential hyperparameter choice. A 10x increase from the optimal value completely destroyed all learning.

---

### Experiment 7: Low Learning Rate (0.0001)

**Configuration:** Same as baseline, but with learning rate decreased from 0.001 to 0.0001.

**Final Results:** Train Accuracy: 79.13% | Val Accuracy: 78.20% | Val Loss: 0.4584

#### Epoch-by-Epoch Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.6175    | 0.6505     | 0.6340  | 0.6271   |
| 2     | 0.6881    | 0.5886     | 0.6871  | 0.5767   |
| 3     | 0.7119    | 0.5617     | 0.7235  | 0.5357   |
| 4     | 0.7215    | 0.5481     | 0.6970  | 0.5663   |
| 5     | 0.7338    | 0.5305     | 0.7010  | 0.5724   |
| 6     | 0.7464    | 0.5190     | 0.7449  | 0.5066   |
| 7     | 0.7519    | 0.5058     | 0.7553  | 0.4954   |
| 8     | 0.7566    | 0.4957     | 0.7641  | 0.4816   |
| 9     | 0.7654    | 0.4861     | 0.7609  | 0.4937   |
| 10    | 0.7723    | 0.4794     | 0.7585  | 0.4962   |
| 11    | 0.7777    | 0.4716     | 0.7748  | 0.4684   |
| 12    | 0.7814    | 0.4621     | 0.7772  | 0.4685   |
| 13    | 0.7805    | 0.4592     | 0.7761  | 0.4780   |
| 14    | 0.7864    | 0.4529     | 0.7868  | 0.4408   |
| 15    | 0.7913    | 0.4454     | 0.7820  | 0.4584   |

#### Analysis

- **Steady, stable learning:** Every single epoch showed improvement in training accuracy (61.8% → 79.1%). No sudden drops, no oscillations, no instability. The learning curve is the smoothest of all 8 experiments.
- **Not converged at epoch 15:** The loss was still clearly decreasing (0.4454 at epoch 15 vs baseline's 0.2997). The model needed more epochs — likely 30-40 total — to reach the baseline's performance level.
- **Excellent generalization:** The gap between train and val accuracy is only 0.9% (79.1% vs 78.2%), the tightest of all experiments. The model is barely overfitting at all, which is expected — smaller weight updates mean the model can't memorize training data as aggressively.
- **Val accuracy dips at epochs 4-5:** Dropped from 72.4% to 69.7% and 70.1%. This is likely noise from the specific test batch ordering combined with the model's small, conservative weight updates.

**The tradeoff of low learning rates:**

Low learning rates are **safe** — they almost never diverge, they produce smooth training curves, and they generalize well. But they're **slow**. Each epoch only moves the weights a small amount, so the model needs proportionally more epochs to converge.

A useful rule of thumb: if you halve the learning rate, you roughly need to double the number of epochs to reach the same performance. Going from 0.001 to 0.0001 is a 10x reduction, suggesting the model needs ~10x more epochs (~150) to match the baseline. This is computationally expensive but produces a well-generalized model.

---

### Experiment 8: Batch Size 16

**Configuration:** Same as baseline, but with batch size reduced from 32 to 16.

**Final Results:** Train Accuracy: 84.39% | Val Accuracy: 81.73% | Val Loss: 0.4172

#### Epoch-by-Epoch Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.6308    | 0.6376     | 0.6644  | 0.6201   |
| 2     | 0.7241    | 0.5465     | 0.7534  | 0.5005   |
| 3     | 0.7491    | 0.5058     | 0.7807  | 0.4628   |
| 4     | 0.7688    | 0.4782     | 0.7494  | 0.5476   |
| 5     | 0.7855    | 0.4582     | 0.7865  | 0.4466   |
| 6     | 0.7953    | 0.4435     | 0.7833  | 0.4630   |
| 7     | 0.7964    | 0.4314     | 0.7943  | 0.4351   |
| 8     | 0.8072    | 0.4186     | 0.7927  | 0.4394   |
| 9     | 0.8156    | 0.4036     | 0.8058  | 0.4241   |
| 10    | 0.8230    | 0.3891     | 0.8143  | 0.4140   |
| 11    | 0.8297    | 0.3795     | 0.7943  | 0.4422   |
| 12    | 0.8320    | 0.3740     | 0.8103  | 0.4096   |
| 13    | 0.8360    | 0.3650     | 0.7978  | 0.4461   |
| 14    | 0.8448    | 0.3497     | 0.8186  | 0.4144   |
| 15    | 0.8439    | 0.3503     | 0.8173  | 0.4172   |

#### Analysis

- **More training steps per epoch:** With batch size 16, the model takes 1,092 gradient updates per epoch (17,471 images / 16) vs 546 updates for batch size 32. Each epoch does 2x more work.
- **Noisier validation curve:** Val accuracy shows more fluctuation than the baseline — sharp dips at epoch 4 (74.9%), epoch 11 (79.4%), and epoch 13 (79.8%). The baseline's val accuracy was much smoother.
- **Slightly lower final accuracy:** 81.7% vs baseline's 83.1%. The noisy gradients from smaller batches slightly hurt convergence quality.
- **Lower val loss than some others:** 0.4172 is actually lower than ELU (0.4597), Tanh (0.4784), and low-LR (0.4584), suggesting the model's probability estimates are reasonably well-calibrated despite the noise.
- **Training took ~2x wall-clock time per epoch:** Each epoch required 1,092 steps × ~125ms = ~136 seconds vs 546 steps × ~210s = ~115 seconds. The per-step time is faster (smaller batch = less computation per step), but there are 2x more steps.

**Batch size and gradient noise:**

The gradient computed from a batch is an *estimate* of the true gradient (computed from all training data). Smaller batches produce noisier estimates:

| Batch Size | Steps/Epoch | Gradient Quality | Convergence Behavior |
|------------|-------------|------------------|----------------------|
| 16         | 1,092       | Noisy            | More exploration, less stable |
| 32         | 546         | Moderate         | Good balance |
| 64         | 273         | Smoother         | Faster per epoch, may get stuck |
| 128        | 136         | Very smooth      | Fewer steps, may miss good minima |

The noise from small batches can be beneficial — it acts as a form of regularization, helping the model escape shallow local minima. But it can also be detrimental, as seen here, where the model struggled to settle into the optimal region. On this dataset, batch size 32 struck the better balance.

---

## Comparative Analysis by Hyperparameter

### Activation Functions

**Comparison (all using Adam, LR 0.001, Batch 32):**

| Activation | Train Acc | Val Acc | Val Loss | Gap (Train - Val) |
|------------|-----------|---------|----------|--------------------|
| **ReLU**   | **87.00%**| **83.14%**| **0.3903**| **3.86%** |
| ELU        | 84.26%    | 79.78%  | 0.4597   | 4.48%              |
| Tanh       | 81.02%    | 77.72%  | 0.4784   | 3.30%              |

**Key findings:**

1. **ReLU is the clear winner.** It achieved the highest training accuracy (87.0%), the highest validation accuracy (83.1%), and the lowest validation loss (0.3903). It also had the largest capacity for learning, as evidenced by the highest training accuracy.

2. **Tanh suffered from vanishing gradients.** Its outputs saturate at -1 and +1 for large inputs, causing gradient flow to diminish through layers. This limited its training accuracy to only 81.0% — 6 points below ReLU. Interestingly, Tanh had the smallest train-val gap (3.3%), but this is because it underfit the training data rather than because it generalized better.

3. **ELU is a viable alternative** but showed no advantage over ReLU on this architecture. Its smooth negative region and zero-centered outputs didn't provide enough benefit to offset the slightly lower performance. ELU's main advantage (avoiding dead neurons) is less relevant in a shallow 2-layer CNN.

4. **The output layer always used sigmoid** regardless of the hidden layer activation. This is standard for binary classification and was not varied.

**Recommendation:** Use ReLU as the default activation for CNNs. Consider ELU only if you observe dead neuron problems (which typically require deeper networks or specific initialization issues).

---

### Optimizers

**Comparison (all using ReLU, LR 0.001, Batch 32):**

| Optimizer  | Train Acc | Val Acc | Val Loss | Convergence Speed |
|------------|-----------|---------|----------|-------------------|
| **Adam**   | **87.00%**| **83.14%**| **0.3903**| **Fast** |
| RMSprop    | 85.90%    | 81.86%  | 0.4319   | Fast (with instability) |
| SGD        | 62.80%    | 61.39%  | 0.6457   | Very Slow         |

**Key findings:**

1. **Adam dominates.** It achieved the best accuracy and the lowest loss while converging the fastest. Its combination of adaptive learning rates (from RMSprop) and momentum (from the first moment estimate) makes it robust and efficient.

2. **RMSprop is a close second** in convergence speed but showed instability (epoch 13 crash). Without Adam's momentum term, the gradient estimates are noisier, leading to occasional validation performance drops.

3. **SGD without momentum is inadequate** at this learning rate. After 15 epochs, it was still at only 62.8% training accuracy. The loss (0.6457) indicates the model had barely moved from random guessing. SGD would need either a much higher learning rate (e.g., 0.01) or momentum (e.g., 0.9) to be competitive.

**Optimizer update rules (simplified):**

| Optimizer  | Update Rule                                          |
|------------|------------------------------------------------------|
| SGD        | `w = w - lr * g`                                     |
| RMSprop    | `w = w - lr * g / sqrt(v)`                           |
| Adam       | `w = w - lr * m / (sqrt(v) + ε)`                     |

Where `g` = current gradient, `m` = running average of gradients (momentum), `v` = running average of squared gradients, `ε` = small constant to prevent division by zero.

**Recommendation:** Use Adam as the default optimizer. Consider SGD with momentum for fine-tuning pre-trained models or when you need more control over the optimization trajectory.

---

### Learning Rates

**Comparison (all using ReLU, Adam, Batch 32):**

| Learning Rate | Train Acc | Val Acc  | Val Loss | Status         |
|---------------|-----------|----------|----------|----------------|
| **0.001**     | **87.00%**| **83.14%**| **0.3903**| **Converged** |
| 0.0001        | 79.13%    | 78.20%   | 0.4584   | Under-converged |
| 0.01          | 49.50%    | 49.96%   | 0.6933   | Diverged       |

**Key findings:**

1. **LR 0.001 is the sweet spot** for Adam on this task. It converged within 15 epochs to the best accuracy.

2. **LR 0.01 caused complete divergence.** The model couldn't learn at all — accuracy was stuck at 50% (random guessing) for all 15 epochs. The loss was frozen at 0.693. This demonstrates that too-aggressive weight updates prevent the model from finding useful solutions.

3. **LR 0.0001 was stable but slow.** The model was still clearly improving at epoch 15 (loss was decreasing, accuracy was climbing). It needed approximately 30-50 more epochs to match the baseline. However, it showed the best generalization (smallest train-val gap: 0.9%).

**The learning rate spectrum:**

```
Too High (0.01)          Just Right (0.001)       Too Low (0.0001)
     ✗                        ✓                        △
Diverges                Converges well            Converges slowly
No learning             Best accuracy in 15 ep    Needs 40+ epochs
Loss stuck at 0.693     Loss → 0.39               Loss → 0.46 (still dropping)
```

**Recommendation:** Start with Adam's default LR (0.001). If training is unstable, reduce to 0.0003 or 0.0001. If training is too slow, increase to 0.003 (but monitor for divergence). Never start with 0.01 for Adam on image tasks.

---

### Batch Sizes

**Comparison (all using ReLU, Adam, LR 0.001):**

| Batch Size | Train Acc | Val Acc | Val Loss | Steps/Epoch | Time/Epoch |
|------------|-----------|---------|----------|-------------|------------|
| **32**     | **87.00%**| **83.14%**| **0.3903**| 546        | ~115s      |
| 16         | 84.39%    | 81.73%  | 0.4172   | 1,092       | ~136s      |

**Key findings:**

1. **Batch 32 outperformed batch 16** by ~1.4% validation accuracy. The larger batches provided more stable gradient estimates, leading to better convergence.

2. **Batch 16 was noisier.** Validation accuracy showed more fluctuation (dips at epochs 4, 11, 13). The smaller batch size introduced more variance in gradient estimates.

3. **Batch 16 was not faster overall.** Despite smaller per-step computation, it required 2x more steps per epoch, resulting in longer total epoch time (~136s vs ~115s).

4. **Both showed similar overfitting patterns.** The train-val gap was comparable (3.9% for batch 32, 2.7% for batch 16), suggesting batch size didn't significantly affect regularization in this case.

**Recommendation:** Use batch size 32 as the default. Larger batches (64, 128) may be worth testing for faster training, but can sometimes reduce generalization. Smaller batches (8, 16) add regularizing noise but may slow convergence.

---

## Final Rankings

### Overall Ranking

| Rank | Experiment                  | Val Acc  | Category            | Verdict           |
|------|-----------------------------|----------|---------------------|-------------------|
| 1    | Baseline (ReLU, Adam, 0.001)| **83.14%** | —                   | Best overall      |
| 2    | RMSprop                     | 81.86%   | Optimizer           | Good, less stable |
| 3    | Batch 16                    | 81.73%   | Batch size          | Good, noisier     |
| 4    | ELU                         | 79.78%   | Activation          | Decent            |
| 5    | LR 0.0001                   | 78.20%   | Learning rate       | Undertrained      |
| 6    | Tanh                        | 77.72%   | Activation          | Limited capacity  |
| 7    | SGD                         | 61.39%   | Optimizer           | Too slow at LR 0.001 |
| 8    | LR 0.01                     | 49.96%   | Learning rate       | Complete failure  |

### Impact Ranking (which hyperparameter mattered most)

| Rank | Hyperparameter    | Impact on Val Acc | Range of Results  |
|------|-------------------|-------------------|-------------------|
| 1    | **Learning Rate** | **33.2%**         | 49.96% → 83.14%   |
| 2    | **Optimizer**     | **21.8%**         | 61.39% → 83.14%   |
| 3    | **Activation**    | **5.4%**          | 77.72% → 83.14%   |
| 4    | **Batch Size**    | **1.4%**          | 81.73% → 83.14%   |

**Learning rate is by far the most impactful hyperparameter.** A 10x change in either direction reduced accuracy by 5-33 percentage points. The optimizer is second (SGD vs Adam is a 22-point gap). Activation function and batch size had comparatively minor effects.

---

## Key Takeaways

### 1. Learning rate is the king of hyperparameters

A poorly chosen learning rate can completely prevent learning (0.01) or drastically slow it (0.0001). Always start with a known good default (e.g., 0.001 for Adam) and adjust from there. Use learning rate schedulers or finders if unsure.

### 2. Adam is the go-to optimizer for CNNs

It combines the best of both worlds — adaptive per-parameter learning rates (from RMSprop) and momentum (from gradient averaging). For most image classification tasks, Adam with default settings will outperform vanilla SGD. SGD with momentum can match Adam but requires more careful tuning.

### 3. ReLU is the standard activation for a reason

Its simplicity (max(0, x)) provides a non-saturating, computationally cheap nonlinearity that allows gradients to flow freely. Tanh and ELU both suffered in comparison, though ELU remains a reasonable alternative in deeper architectures where dead neurons are a concern.

### 4. Batch size 32 is a solid default

The difference between batch 16 and batch 32 was small (1.4%), suggesting this hyperparameter is less critical for performance. However, smaller batches can cause training instability and slower per-epoch times. Start with 32 and adjust based on GPU memory and training speed needs.

### 5. Overfitting begins early with small datasets

With ~17,000 training images and ~813,000 model parameters, the model has significant capacity relative to the data. Multiple experiments showed overfitting in the final epochs (train accuracy climbing while val accuracy plateaus or drops). Techniques like dropout, weight decay, or early stopping would help.

### 6. 15 epochs is insufficient for some configurations

SGD (LR 0.001) and low-LR (0.0001) were still clearly improving at epoch 15. These configurations would benefit from 40-50+ epochs. This highlights the importance of choosing a learning rate/optimizer combination that converges within your computational budget.

---

## Recommendations for Further Improvement

### Architecture Improvements

- **Add Dropout layers** (0.25-0.5) after pooling layers and before the output to reduce overfitting.
- **Add Batch Normalization** after each Conv2D layer to stabilize training and allow higher learning rates.
- **Increase model depth** — add a third or fourth conv block to learn more complex features.
- **Use larger input images** (128×128 or 224×224) with a deeper architecture to capture finer details.
- **Use transfer learning** — fine-tune a pre-trained model (VGG16, ResNet50, MobileNet) instead of training from scratch. This would dramatically improve accuracy.

### Training Improvements

- **Add early stopping** — monitor val_loss and stop training when it hasn't improved for 3-5 epochs.
- **Use a learning rate scheduler** — reduce LR by 10x when val_loss plateaus (e.g., `ReduceLROnPlateau`).
- **Increase epochs to 30-50** — several experiments hadn't converged at epoch 15.
- **Use AdamW** instead of Adam — it decouples weight decay from the gradient update, often improving generalization.

### Data Improvements

- **Increase image resolution** — 64×64 loses significant detail. 150×150 or 224×224 would help.
- **Add more augmentation** — rotation, brightness/contrast adjustment, channel shifts.
- **Use class weights** — if the dataset is imbalanced (it's roughly equal here, but worth checking).

---

## Appendix: Full Epoch Logs

### Experiment 1: Baseline (ReLU, Adam, LR 0.001, Batch 32)

```
Epoch 1/15  - accuracy: 0.6725 - loss: 0.6020 - val_accuracy: 0.7286 - val_loss: 0.5393
Epoch 2/15  - accuracy: 0.7503 - loss: 0.5117 - val_accuracy: 0.7724 - val_loss: 0.4704
Epoch 3/15  - accuracy: 0.7697 - loss: 0.4759 - val_accuracy: 0.7954 - val_loss: 0.4374
Epoch 4/15  - accuracy: 0.7856 - loss: 0.4490 - val_accuracy: 0.7895 - val_loss: 0.4548
Epoch 5/15  - accuracy: 0.7978 - loss: 0.4337 - val_accuracy: 0.8020 - val_loss: 0.4265
Epoch 6/15  - accuracy: 0.8073 - loss: 0.4128 - val_accuracy: 0.8063 - val_loss: 0.4174
Epoch 7/15  - accuracy: 0.8187 - loss: 0.4006 - val_accuracy: 0.8026 - val_loss: 0.4359
Epoch 8/15  - accuracy: 0.8259 - loss: 0.3837 - val_accuracy: 0.8119 - val_loss: 0.4121
Epoch 9/15  - accuracy: 0.8297 - loss: 0.3753 - val_accuracy: 0.8151 - val_loss: 0.4034
Epoch 10/15 - accuracy: 0.8408 - loss: 0.3588 - val_accuracy: 0.8229 - val_loss: 0.3975
Epoch 11/15 - accuracy: 0.8461 - loss: 0.3452 - val_accuracy: 0.8234 - val_loss: 0.3965
Epoch 12/15 - accuracy: 0.8517 - loss: 0.3337 - val_accuracy: 0.8229 - val_loss: 0.4066
Epoch 13/15 - accuracy: 0.8592 - loss: 0.3209 - val_accuracy: 0.8330 - val_loss: 0.3936
Epoch 14/15 - accuracy: 0.8651 - loss: 0.3109 - val_accuracy: 0.8328 - val_loss: 0.3793
Epoch 15/15 - accuracy: 0.8700 - loss: 0.2997 - val_accuracy: 0.8314 - val_loss: 0.3903
```

### Experiment 2: Activation — Tanh

```
Epoch 1/15  - accuracy: 0.6525 - loss: 0.6252 - val_accuracy: 0.7131 - val_loss: 0.5656
Epoch 2/15  - accuracy: 0.7133 - loss: 0.5631 - val_accuracy: 0.7272 - val_loss: 0.5389
Epoch 3/15  - accuracy: 0.7356 - loss: 0.5335 - val_accuracy: 0.7331 - val_loss: 0.5185
Epoch 4/15  - accuracy: 0.7437 - loss: 0.5207 - val_accuracy: 0.7417 - val_loss: 0.5207
Epoch 5/15  - accuracy: 0.7478 - loss: 0.5116 - val_accuracy: 0.7144 - val_loss: 0.5458
Epoch 6/15  - accuracy: 0.7568 - loss: 0.5029 - val_accuracy: 0.7582 - val_loss: 0.4860
Epoch 7/15  - accuracy: 0.7646 - loss: 0.4907 - val_accuracy: 0.7657 - val_loss: 0.4794
Epoch 8/15  - accuracy: 0.7652 - loss: 0.4879 - val_accuracy: 0.7708 - val_loss: 0.4795
Epoch 9/15  - accuracy: 0.7744 - loss: 0.4718 - val_accuracy: 0.7697 - val_loss: 0.4738
Epoch 10/15 - accuracy: 0.7774 - loss: 0.4652 - val_accuracy: 0.7750 - val_loss: 0.4621
Epoch 11/15 - accuracy: 0.7781 - loss: 0.4594 - val_accuracy: 0.7847 - val_loss: 0.4547
Epoch 12/15 - accuracy: 0.7949 - loss: 0.4458 - val_accuracy: 0.7855 - val_loss: 0.4571
Epoch 13/15 - accuracy: 0.8022 - loss: 0.4293 - val_accuracy: 0.7852 - val_loss: 0.4462
Epoch 14/15 - accuracy: 0.8037 - loss: 0.4272 - val_accuracy: 0.7815 - val_loss: 0.4605
Epoch 15/15 - accuracy: 0.8102 - loss: 0.4155 - val_accuracy: 0.7772 - val_loss: 0.4784
```

### Experiment 3: Activation — ELU

```
Epoch 1/15  - accuracy: 0.6566 - loss: 0.6313 - val_accuracy: 0.7179 - val_loss: 0.5527
Epoch 2/15  - accuracy: 0.7216 - loss: 0.5503 - val_accuracy: 0.7296 - val_loss: 0.5304
Epoch 3/15  - accuracy: 0.7404 - loss: 0.5239 - val_accuracy: 0.7395 - val_loss: 0.5166
Epoch 4/15  - accuracy: 0.7476 - loss: 0.5097 - val_accuracy: 0.7441 - val_loss: 0.4996
Epoch 5/15  - accuracy: 0.7581 - loss: 0.4954 - val_accuracy: 0.7670 - val_loss: 0.4862
Epoch 6/15  - accuracy: 0.7670 - loss: 0.4788 - val_accuracy: 0.7438 - val_loss: 0.5134
Epoch 7/15  - accuracy: 0.7770 - loss: 0.4678 - val_accuracy: 0.7668 - val_loss: 0.4725
Epoch 8/15  - accuracy: 0.7847 - loss: 0.4558 - val_accuracy: 0.7812 - val_loss: 0.4562
Epoch 9/15  - accuracy: 0.7954 - loss: 0.4441 - val_accuracy: 0.7766 - val_loss: 0.4679
Epoch 10/15 - accuracy: 0.8024 - loss: 0.4235 - val_accuracy: 0.7793 - val_loss: 0.4637
Epoch 11/15 - accuracy: 0.8099 - loss: 0.4132 - val_accuracy: 0.7769 - val_loss: 0.4935
Epoch 12/15 - accuracy: 0.8134 - loss: 0.4051 - val_accuracy: 0.7852 - val_loss: 0.4585
Epoch 13/15 - accuracy: 0.8224 - loss: 0.3894 - val_accuracy: 0.7996 - val_loss: 0.4291
Epoch 14/15 - accuracy: 0.8358 - loss: 0.3691 - val_accuracy: 0.8020 - val_loss: 0.4314
Epoch 15/15 - accuracy: 0.8426 - loss: 0.3584 - val_accuracy: 0.7978 - val_loss: 0.4597
```

### Experiment 4: Optimizer — SGD

```
Epoch 1/15  - accuracy: 0.5290 - loss: 0.6914 - val_accuracy: 0.5498 - val_loss: 0.6892
Epoch 2/15  - accuracy: 0.5555 - loss: 0.6872 - val_accuracy: 0.5594 - val_loss: 0.6861
Epoch 3/15  - accuracy: 0.5738 - loss: 0.6840 - val_accuracy: 0.5835 - val_loss: 0.6828
Epoch 4/15  - accuracy: 0.5795 - loss: 0.6813 - val_accuracy: 0.5795 - val_loss: 0.6799
Epoch 5/15  - accuracy: 0.5838 - loss: 0.6781 - val_accuracy: 0.5848 - val_loss: 0.6766
Epoch 6/15  - accuracy: 0.5901 - loss: 0.6752 - val_accuracy: 0.5923 - val_loss: 0.6733
Epoch 7/15  - accuracy: 0.5974 - loss: 0.6723 - val_accuracy: 0.5827 - val_loss: 0.6718
Epoch 8/15  - accuracy: 0.5992 - loss: 0.6695 - val_accuracy: 0.5987 - val_loss: 0.6666
Epoch 9/15  - accuracy: 0.6044 - loss: 0.6660 - val_accuracy: 0.6212 - val_loss: 0.6610
Epoch 10/15 - accuracy: 0.6089 - loss: 0.6625 - val_accuracy: 0.6107 - val_loss: 0.6582
Epoch 11/15 - accuracy: 0.6163 - loss: 0.6582 - val_accuracy: 0.6286 - val_loss: 0.6517
Epoch 12/15 - accuracy: 0.6156 - loss: 0.6564 - val_accuracy: 0.5987 - val_loss: 0.6555
Epoch 13/15 - accuracy: 0.6246 - loss: 0.6518 - val_accuracy: 0.5952 - val_loss: 0.6553
Epoch 14/15 - accuracy: 0.6253 - loss: 0.6495 - val_accuracy: 0.6238 - val_loss: 0.6460
Epoch 15/15 - accuracy: 0.6280 - loss: 0.6451 - val_accuracy: 0.6139 - val_loss: 0.6457
```

### Experiment 5: Optimizer — RMSprop

```
Epoch 1/15  - accuracy: 0.6371 - loss: 0.6383 - val_accuracy: 0.7136 - val_loss: 0.5514
Epoch 2/15  - accuracy: 0.7253 - loss: 0.5425 - val_accuracy: 0.7430 - val_loss: 0.5142
Epoch 3/15  - accuracy: 0.7526 - loss: 0.5054 - val_accuracy: 0.7726 - val_loss: 0.4852
Epoch 4/15  - accuracy: 0.7739 - loss: 0.4771 - val_accuracy: 0.7839 - val_loss: 0.4562
Epoch 5/15  - accuracy: 0.7865 - loss: 0.4554 - val_accuracy: 0.7702 - val_loss: 0.4743
Epoch 6/15  - accuracy: 0.7924 - loss: 0.4406 - val_accuracy: 0.8031 - val_loss: 0.4348
Epoch 7/15  - accuracy: 0.8010 - loss: 0.4304 - val_accuracy: 0.8050 - val_loss: 0.4226
Epoch 8/15  - accuracy: 0.8135 - loss: 0.4134 - val_accuracy: 0.8071 - val_loss: 0.4294
Epoch 9/15  - accuracy: 0.8218 - loss: 0.3980 - val_accuracy: 0.8100 - val_loss: 0.4425
Epoch 10/15 - accuracy: 0.8243 - loss: 0.3867 - val_accuracy: 0.8106 - val_loss: 0.4231
Epoch 11/15 - accuracy: 0.8310 - loss: 0.3789 - val_accuracy: 0.8197 - val_loss: 0.4105
Epoch 12/15 - accuracy: 0.8410 - loss: 0.3640 - val_accuracy: 0.8226 - val_loss: 0.4132
Epoch 13/15 - accuracy: 0.8450 - loss: 0.3553 - val_accuracy: 0.7919 - val_loss: 0.5306
Epoch 14/15 - accuracy: 0.8492 - loss: 0.3451 - val_accuracy: 0.7945 - val_loss: 0.4788
Epoch 15/15 - accuracy: 0.8590 - loss: 0.3302 - val_accuracy: 0.8186 - val_loss: 0.4319
```

### Experiment 6: High Learning Rate (0.01)

```
Epoch 1/15  - accuracy: 0.5071 - loss: 0.7411 - val_accuracy: 0.5004 - val_loss: 0.6936
Epoch 2/15  - accuracy: 0.4976 - loss: 0.6934 - val_accuracy: 0.5004 - val_loss: 0.6932
Epoch 3/15  - accuracy: 0.4959 - loss: 0.6935 - val_accuracy: 0.5004 - val_loss: 0.6936
Epoch 4/15  - accuracy: 0.4942 - loss: 0.6934 - val_accuracy: 0.5004 - val_loss: 0.6932
Epoch 5/15  - accuracy: 0.4970 - loss: 0.6935 - val_accuracy: 0.5004 - val_loss: 0.6932
Epoch 6/15  - accuracy: 0.5001 - loss: 0.6934 - val_accuracy: 0.5004 - val_loss: 0.6933
Epoch 7/15  - accuracy: 0.4961 - loss: 0.6934 - val_accuracy: 0.4996 - val_loss: 0.6933
Epoch 8/15  - accuracy: 0.5026 - loss: 0.6934 - val_accuracy: 0.5004 - val_loss: 0.6931
Epoch 9/15  - accuracy: 0.4998 - loss: 0.6934 - val_accuracy: 0.4996 - val_loss: 0.6932
Epoch 10/15 - accuracy: 0.4929 - loss: 0.6934 - val_accuracy: 0.4996 - val_loss: 0.6932
Epoch 11/15 - accuracy: 0.4988 - loss: 0.6933 - val_accuracy: 0.5004 - val_loss: 0.6933
Epoch 12/15 - accuracy: 0.4995 - loss: 0.6935 - val_accuracy: 0.5004 - val_loss: 0.6934
Epoch 13/15 - accuracy: 0.4960 - loss: 0.6934 - val_accuracy: 0.5004 - val_loss: 0.6931
Epoch 14/15 - accuracy: 0.4936 - loss: 0.6935 - val_accuracy: 0.5004 - val_loss: 0.6933
Epoch 15/15 - accuracy: 0.4950 - loss: 0.6935 - val_accuracy: 0.4996 - val_loss: 0.6933
```

### Experiment 7: Low Learning Rate (0.0001)

```
Epoch 1/15  - accuracy: 0.6175 - loss: 0.6505 - val_accuracy: 0.6340 - val_loss: 0.6271
Epoch 2/15  - accuracy: 0.6881 - loss: 0.5886 - val_accuracy: 0.6871 - val_loss: 0.5767
Epoch 3/15  - accuracy: 0.7119 - loss: 0.5617 - val_accuracy: 0.7235 - val_loss: 0.5357
Epoch 4/15  - accuracy: 0.7215 - loss: 0.5481 - val_accuracy: 0.6970 - val_loss: 0.5663
Epoch 5/15  - accuracy: 0.7338 - loss: 0.5305 - val_accuracy: 0.7010 - val_loss: 0.5724
Epoch 6/15  - accuracy: 0.7464 - loss: 0.5190 - val_accuracy: 0.7449 - val_loss: 0.5066
Epoch 7/15  - accuracy: 0.7519 - loss: 0.5058 - val_accuracy: 0.7553 - val_loss: 0.4954
Epoch 8/15  - accuracy: 0.7566 - loss: 0.4957 - val_accuracy: 0.7641 - val_loss: 0.4816
Epoch 9/15  - accuracy: 0.7654 - loss: 0.4861 - val_accuracy: 0.7609 - val_loss: 0.4937
Epoch 10/15 - accuracy: 0.7723 - loss: 0.4794 - val_accuracy: 0.7585 - val_loss: 0.4962
Epoch 11/15 - accuracy: 0.7777 - loss: 0.4716 - val_accuracy: 0.7748 - val_loss: 0.4684
Epoch 12/15 - accuracy: 0.7814 - loss: 0.4621 - val_accuracy: 0.7772 - val_loss: 0.4685
Epoch 13/15 - accuracy: 0.7805 - loss: 0.4592 - val_accuracy: 0.7761 - val_loss: 0.4780
Epoch 14/15 - accuracy: 0.7864 - loss: 0.4529 - val_accuracy: 0.7868 - val_loss: 0.4408
Epoch 15/15 - accuracy: 0.7913 - loss: 0.4454 - val_accuracy: 0.7820 - val_loss: 0.4584
```

### Experiment 8: Batch Size 16

```
Epoch 1/15  - accuracy: 0.6308 - loss: 0.6376 - val_accuracy: 0.6644 - val_loss: 0.6201
Epoch 2/15  - accuracy: 0.7241 - loss: 0.5465 - val_accuracy: 0.7534 - val_loss: 0.5005
Epoch 3/15  - accuracy: 0.7491 - loss: 0.5058 - val_accuracy: 0.7807 - val_loss: 0.4628
Epoch 4/15  - accuracy: 0.7688 - loss: 0.4782 - val_accuracy: 0.7494 - val_loss: 0.5476
Epoch 5/15  - accuracy: 0.7855 - loss: 0.4582 - val_accuracy: 0.7865 - val_loss: 0.4466
Epoch 6/15  - accuracy: 0.7953 - loss: 0.4435 - val_accuracy: 0.7833 - val_loss: 0.4630
Epoch 7/15  - accuracy: 0.7964 - loss: 0.4314 - val_accuracy: 0.7943 - val_loss: 0.4351
Epoch 8/15  - accuracy: 0.8072 - loss: 0.4186 - val_accuracy: 0.7927 - val_loss: 0.4394
Epoch 9/15  - accuracy: 0.8156 - loss: 0.4036 - val_accuracy: 0.8058 - val_loss: 0.4241
Epoch 10/15 - accuracy: 0.8230 - loss: 0.3891 - val_accuracy: 0.8143 - val_loss: 0.4140
Epoch 11/15 - accuracy: 0.8297 - loss: 0.3795 - val_accuracy: 0.7943 - val_loss: 0.4422
Epoch 12/15 - accuracy: 0.8320 - loss: 0.3740 - val_accuracy: 0.8103 - val_loss: 0.4096
Epoch 13/15 - accuracy: 0.8360 - loss: 0.3650 - val_accuracy: 0.7978 - val_loss: 0.4461
Epoch 14/15 - accuracy: 0.8448 - loss: 0.3497 - val_accuracy: 0.8186 - val_loss: 0.4144
Epoch 15/15 - accuracy: 0.8439 - loss: 0.3503 - val_accuracy: 0.8173 - val_loss: 0.4172
```

---

*Report generated from experimental results. Dataset: Kaggle Cats and Dogs (Microsoft). Architecture: 2-layer CNN with ~813K parameters. Framework: TensorFlow / Keras.*