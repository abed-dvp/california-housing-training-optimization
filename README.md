# California Housing Training Optimization

## Project Goal

Understand what happens behind `.fit()` by progressively implementing and comparing optimization methods and loss functions on the California Housing dataset. This project demystifies machine learning estimators by taking an educational, bottom-up journey from raw calculus and manual Gradient Descent to production first- and second-order solvers.

## Dataset

**California Housing** (`sklearn.datasets.fetch_california_housing`)

- **Observations**: 20,640 block groups (16,512 train / 4,128 test)
- **Input Features**: 8 numerical predictors (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`)
- **Continuous Target**: `MedHouseVal` (Median house value expressed in $100,000s)
- **Data Integrity**: Clean numeric dataset with 0 missing values and 0 duplicate rows

## What This Project Demonstrates

- **Learned Model Parameters**: Deconstructing what `.fit()` learns ($\beta_0$ intercept and $\beta_j$ feature weights).
- **Gradient Descent from Scratch**: Deriving analytical partial derivatives and parameter update rules for univariate and multivariate models.
- **Learning Rate and Convergence**: Exploring convergence thresholds, minimum step sizes, and the dynamics of overshooting vs. slow convergence.
- **Feature Scaling and Optimization**: Demonstrating why unscaled multi-feature GD diverges while standardized features converge smoothly.
- **Loss Landscapes and Contours**: Visualizing 2D and multi-dimensional energy bowls and tracing optimization trajectories.
- **OLS Geometry**: Validating the fundamental linear algebra property of Ordinary Least Squares — residual orthogonality to the column space ($\mathbf{X}^T \mathbf{e} \approx \mathbf{0}$).
- **Solver Variants**: Contrasting update frequencies and gradient noise across Batch GD ($B=N$), Mini-Batch GD ($B=32$), and Stochastic GD ($B=1$).
- **Direct Solvers vs. Iterative Solvers**: Benchmarking exact numerical least-squares solutions against iterative stochastic optimization.
- **Regression Loss Functions**: Evaluating how $L_1$ (MAE), $L_2$ (MSE), and Huber loss ($\delta=1.35$) penalize residual tails and influence learned parameters.
- **Classification Loss Functions**: Transforming linear scores via the Sigmoid function to model probabilities, calculating Likelihood and Log-Likelihood, minimizing Log Loss, and comparing with margin-based Hinge Loss.
- **Second-Order Optimization Concepts**: Understanding curvature via the Hessian matrix, and comparing Newton-style (`newton-cg`) vs. quasi-Newton (`lbfgs`) solvers.
- **Loss Functions vs. Performance Metrics**: Clarifying the distinct roles of training optimization objectives vs. post-training evaluation metrics.
- **Parameters vs. Hyperparameters**: Differentiating internal values learned from data from external configurations chosen before training.

## Project Structure / Learning Flow

The project follows a rigorous 8-step curriculum-locked progression:

1. **Step 1: Behind `.fit()`** — Baseline 1-feature OLS, manual hypothesis function $h(x, \beta)$, residual calculation, and squared loss bowl exploration.
2. **Step 2: Manual Gradient Descent** — Analytical derivatives, step-by-step parameter updates, learning rate sensitivity ($\eta \in \{0.01, 0.20, 1.05\}$), and multivariate GD.
3. **Step 3: Scaling Effect, Loss Landscape, and OLS Geometry** — 2D loss surface contour mapping, raw divergence vs. standardized convergence, and geometric verification of column-space orthogonality.
4. **Step 4: Batch, Mini-Batch, SGD, and Estimator Benchmarks** — Implementing and diagnosing update frequencies, single-observation gradient noise, and comparing `LinearRegression` with `SGDRegressor`.
5. **Step 5: Regression Loss Functions (L1, L2, Huber)** — Mathematical definitions, residual distribution analysis, extreme error penalties, and training `SGDRegressor` with squared error vs. Huber loss.
6. **Step 6: Classification Loss Functions (Log Loss & Hinge)** — Educational median split binary target, Sigmoid hypothesis, Likelihood product, Log-Likelihood sum, Cross-Entropy minimization, confident wrong prediction penalties, and `SGDClassifier` (Log Loss vs. Hinge).
7. **Step 7: Enhanced GD and Second-Order Methods** — Conceptual intuition for Momentum, AdaGrad, RMSProp, and Adam; Hessian curvature matrices; and comparing `LogisticRegression(solver="newton-cg")` with `LogisticRegression(solver="lbfgs")`.
8. **Step 8: Final Training Synthesis and Project Wrap-up** — Unifying Hypothesis vs. Loss vs. Solver, Parameters vs. Hyperparameters, Loss vs. Metric, and full 56/56 lesson coverage audit.

## Key Experiments

- **Manual GD vs. OLS**: Manual Gradient Descent on raw `MedInc` converged to $\beta_0 = 0.4437$ and $\beta_1 = 0.4195$, matching scikit-learn's analytical solution within $< 0.001$.
- **Feature Standardization**: Unscaled 3-feature GD diverged rapidly within 15 epochs, whereas standardized features converged smoothly to MSE $= 0.6558$.
- **Solver Granularity**: Across 15 epochs on 16,512 samples, Batch GD executed 15 updates, Mini-Batch (32) executed 7,740 updates, and SGD executed 247,680 updates, revealing the trade-off between deterministic steps and gradient noise.
- **Regression Loss Trade-Off**: `SGDRegressor(loss="huber")` achieved lower test MAE (`0.5276` vs `0.5299`) due to linear penalty tails, while `SGDRegressor(loss="squared_error")` achieved lower test RMSE (`0.7420` vs `0.7457`) by penalizing errors quadratically.
- **Classification Objectives**: `SGDClassifier(loss="log_loss")` and `SGDClassifier(loss="hinge")` learned distinct parameter vectors ($\text{mean } |\Delta \beta| = 0.7046$) on identical data, demonstrating how the loss surface reshapes the decision boundary.
- **Second-Order Convergence**: `LogisticRegression(solver="newton-cg")` converged in 6 iterations, while `LogisticRegression(solver="lbfgs")` converged in 25 iterations; both reached virtually identical logistic solutions and produced 100% identical test predictions.

## Core Lesson

Every supervised estimator in machine learning embodies this explicit training chain:

$$\text{Hypothesis Function} \longrightarrow \text{Loss Function} \longrightarrow \text{Solver} \longrightarrow \text{Hyperparameters} \longrightarrow \boldsymbol{.fit(X, y)} \longrightarrow \text{Learned Parameters } \boldsymbol{\beta} \longrightarrow \text{Evaluation Metrics}$$

- **Hypothesis**: Mathematical form connecting inputs to outputs ($h(X, \beta)$).
- **Loss Function**: Mathematical objective surface minimized during training ($L(y, \hat{y})$).
- **Solver**: Algorithmic procedure navigating the loss surface to locate optimal parameters.
- **Hyperparameters**: Configuration settings set before training that govern optimization.
- **Parameters**: Internal weights learned automatically from data during `.fit()`.
- **Metrics**: Post-training domain summaries evaluating practical utility.

## Scope

This is an educational optimization and training mechanics project. To maintain focus on the core optimization curriculum, it intentionally does not include:
- Production API deployment or cloud hosting
- Automated hyperparameter optimization (GridSearchCV / RandomizedSearchCV)
- Automated pipelines (sklearn Pipeline)
- Regularization tuning (Ridge, Lasso, ElasticNet)
- Non-linear models, tree ensembles, or neural networks
- Probability threshold tuning or ROC / PR curve analysis

## How to Run

Clone the repository and install the minimal dependencies:

```bash
git clone https://github.com/abed-dvp/california-housing-training-optimization.git
cd california-housing-training-optimization
pip install -r requirements.txt
```

Launch the interactive Jupyter notebook:

```bash
jupyter notebook notebook.ipynb
```

## Status

Completed — 56 / 56 lesson concepts implemented
