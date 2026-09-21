# California Housing ? Training Mechanics & Optimization from First Principles

A from-first-principles investigation into machine learning optimization, deconstructing what happens behind `.fit()` through manual gradient descent, analytical derivatives, loss landscape geometry, and second-order numerical solvers.

---

## Executive Summary & Core Concept

In applied machine learning, library abstractions like `estimator.fit(X, y)` hide the core mathematical mechanisms that govern model convergence, stability, and parameter estimation.

This case study investigates model training from first principles:
- Deriving analytical loss gradients and executing manual Gradient Descent.
- Examining the geometry of loss landscapes, learning rates, and feature scaling.
- Verifying the fundamental geometric property of Ordinary Least Squares: residual orthogonality ($\\mathbf{X}^T \\mathbf{e} \\approx \\mathbf{0}$).
- Benchmarking Batch vs. Mini-Batch vs. Stochastic Gradient Descent.
- Comparing regression loss functions ($L_1$, $L_2$, Huber) and classification loss functions (Log Loss vs. Hinge).
- Investigating second-order curvature via the Hessian matrix, contrasting Newton-CG and quasi-Newton (L-BFGS) solvers.

Every supervised estimator embodies this explicit training pipeline:

$$\\text{Hypothesis Function } h(X, \\beta) \\longrightarrow \\text{Loss Objective } L(y, \\hat{y}) \\longrightarrow \\text{Solver} \\longrightarrow \\text{Optimization Steps} \\longrightarrow \\text{Learned Weights } \\boldsymbol{\\beta}$$

---

## Dataset

**California Housing** (`sklearn.datasets.fetch_california_housing`):
- **Observations**: 20,640 block groups (16,512 train / 4,128 test)
- **Features**: 8 numeric predictors (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`)
- **Continuous Target**: `MedHouseVal` (Median house value in $100,000s)
- **Data Integrity**: Clean numeric dataset with 0 missing values and 0 duplicates

---

## First-Principles Experiments & Findings

### 1. Manual Gradient Descent vs. OLS Analytical Solution
- Derived analytical partial derivatives for univariate linear regression:
  $$\\frac{\\partial \\text{MSE}}{\\partial \\beta_0} = -\\frac{2}{N}\\sum (y_i - \\hat{y}_i), \\quad \\frac{\\partial \\text{MSE}}{\\partial \\beta_1} = -\\frac{2}{N}\\sum x_i (y_i - \\hat{y}_i)$$
- Manual Gradient Descent on raw `MedInc` converged to $\\beta_0 = 0.4437$ and $\\beta_1 = 0.4195$, matching scikit-learn's analytical least-squares solution within $< 0.001$.

### 2. Feature Standardization & Loss Landscapes
- **Unscaled Multi-Feature GD**: Diverged rapidly within 15 epochs due to anisotropic loss surface curvature (ill-conditioned Hessian).
- **Standardized GD**: Standardizing predictors sphericalized the loss surface, enabling smooth, stable convergence to MSE $= 0.6558$.

### 3. OLS Geometry & Residual Orthogonality
- Empirically validated the foundational linear algebra theorem of least-squares projection: the residual error vector $\\mathbf{e} = \\mathbf{y} - \\hat{\\mathbf{y}}$ is strictly orthogonal to the column space of the feature matrix $\\mathbf{X}$:
  $$\\mathbf{X}^T \\mathbf{e} = \\mathbf{0} \\quad (\\text{evaluated to } < 10^{-12})$$

### 4. Solver Granularity: Batch vs. Mini-Batch vs. Stochastic GD
Across 15 epochs on 16,512 training observations:
- **Batch GD ($B=N$)**: Executed 15 deterministic updates with smooth monotonic loss decrease.
- **Mini-Batch GD ($B=32$)**: Executed 7,740 updates, balancing computational vectorization with gradient exploration.
- **Stochastic GD ($B=1$)**: Executed 247,680 updates, introducing stochastic noise that helps escape saddle points at the cost of high step variance.

### 5. Regression Loss Functions ($L_1$, $L_2$, Huber)
- Evaluated tail sensitivity across loss formulations:
  - `SGDRegressor(loss="squared_error")` penalizes large errors quadratically, yielding lower Test RMSE (`0.7420` vs `0.7457`).
  - `SGDRegressor(loss="huber")` transitions from quadratic to linear penalty beyond threshold $\\delta=1.35$, resisting outliers and yielding lower Test MAE (`0.5276` vs `0.5299`).

### 6. Classification Objectives: Log Loss vs. Hinge Loss
- Mapped regression targets to a binary threshold to contrast probabilistic vs. margin-based classification:
  - `SGDClassifier(loss="log_loss")` minimizes negative log-likelihood, producing calibrated posterior probabilities.
  - `SGDClassifier(loss="hinge")` maximizes classification margin (linear SVM), focusing entirely on boundary-violating points.
  - Learned parameter vectors diverged substantially (mean difference $|\\Delta \\beta| = 0.7046$), proving how the choice of loss function fundamentally alters the decision hyperplane.

### 7. Second-Order Optimization: Newton-CG vs. L-BFGS
- Analyzed second-order curvature using the Hessian matrix $\\mathbf{H} = \\nabla^2 L(\\beta)$:
  - `LogisticRegression(solver="newton-cg")` uses exact conjugate-gradient approximations of the Hessian, converging in **6 iterations**.
  - `LogisticRegression(solver="lbfgs")` approximates the inverse Hessian via gradient histories, converging in **25 iterations**.
  - Both solvers reached identical optimal parameters and 100% identical test predictions, illustrating the trade-off between per-iteration computation and step count.

---

## Running Locally

```bash
git clone https://github.com/abed-dvp/california-housing-training-optimization.git
cd california-housing-training-optimization

pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## Project Context
This case study explores foundational optimization mechanics, parameter estimation, and numerical solvers in machine learning.
