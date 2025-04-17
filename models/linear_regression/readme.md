# 📘 Linear Regression - Complete Model Overview

This section explains the full structure, loss function, and optimizer used in linear regression — the most fundamental supervised learning model in machine learning.

---

## 📐 1. Model Structure

Linear regression assumes a linear relationship between the input features and the output target:

\[
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b
\]

Where:
- \( \mathbf{x} \): input feature vector
- \( \mathbf{w} \): weight vector (model parameters)
- \( b \): bias (intercept)
- \( \hat{y} \): predicted value

✅ The model learns \( \mathbf{w} \) and \( b \) by minimizing the prediction error across the training data.

---

## 📉 2. Loss Function - Mean Squared Error (MSE)

To train a linear regression model, we define a **loss function** that tells us how bad our predictions are.

Most common for regression: **Mean Squared Error (MSE)**

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Where:
- \( y_i \) is the ground truth label
- \( \hat{y}_i \) is the model prediction

🎯 This loss penalizes large errors more heavily, making the model try to fit all points tightly.

---

## ⚙️ 3. Optimizer - Gradient Descent

To minimize the loss function and find the best parameters, we use an **optimization algorithm**. The most fundamental one is **Gradient Descent**.

### 🔄 Update Rule:

\[
w := w - \eta \cdot \frac{\partial \text{Loss}}{\partial w}
\]

Where:
- \( \eta \) is the learning rate
- \( \frac{\partial \text{Loss}}{\partial w} \) is the gradient (partial derivative)

### 🧪 Variants of Gradient Descent:
| Method               | Description                                 |
|----------------------|---------------------------------------------|
| Batch Gradient Descent | Use the entire dataset for each step        |
| Stochastic Gradient Descent (SGD) | Use a single example per update       |
| Mini-batch Gradient Descent | Use a small batch (e.g., 32 samples)   |

---

## 🧪 4. Python Demo (Closed-form solution)

Here is a minimal example of solving linear regression using the **normal equation** (analytical solution):

```python
import numpy as np

# Generate synthetic data
X = np.random.rand(100, 1)
y = 3 * X[:, 0] + 5 + np.random.randn(100) * 0.1  # y = 3x + 5 + noise

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]  # shape: (100, 2)

# Closed-form solution (Normal Equation)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Learned parameters:", theta_best)  # Should be close to [5, 3]
```

This demo learns the true parameters from noisy data.

---

## 📊 5. Visualizing Training (optional Jupyter notebook)

To better understand how gradient descent converges:
- Plot data points vs predicted line
- Animate loss reduction over epochs
- Show how weights evolve step-by-step

(*Coming soon: `visualize_training.ipynb`*)

---

## 🧠 Summary

| Component       | Linear Regression Viewpoint                 |
|----------------|----------------------------------------------|
| Model           | \( \hat{y} = \mathbf{w}^T \mathbf{x} + b \) |
| Loss Function   | Mean Squared Error (MSE)                    |
| Optimizer       | Gradient Descent or Normal Equation         |
| Output          | Continuous numeric prediction               |

This is the starting point of all supervised learning — mastering it lays the foundation for understanding deeper models like logistic regression, trees, and neural nets.

---

© 2025 Pengyi. Part of the [ML-Architectures](../README.md) project.
