## Contributions

# Part 1: Probability Distributions (EM Algorithm)
- Loaded and preprocessed the Galton Families height dataset
- Selected father-child height pairing (per assignment's either/or requirement)
- Implemented deduplication to prevent family-size bias in parent heights
- Implemented the EM algorithm from scratch:
  - Gaussian PDF function
  - E-step (responsibility calculation)
  - M-step (parameter updates)
  - Log-likelihood convergence tracking
- Built the required iteration tracking table (iterations 0, 1, 2)
- Implemented live posterior-probability classification demo
- Validated EM's blind estimates against true known group means

# Part 2: Bayesian Probability

This notebook implements **Bayes' Theorem** using the IMDb Movie Reviews dataset to calculate the posterior probability of positive sentiment based on selected keywords.

## File

- `formative_3_bayesian_probability.ipynb`

## How to Run

1. Open the notebook in Jupyter Notebook, VS Code, or Google Colab.
2. Ensure the IMDb Movie Reviews dataset is available in the expected location.
3. Run all cells from top to bottom (**Run All**).

## What the Notebook Does

- Loads and preprocesses the IMDb Movie Reviews dataset.
- Uses the selected positive keywords:
  - `great`
  - `excellent`
  - `amazing`
  - `love`
- Calculates:
  - Prior Probability: `P(Positive)`
  - Likelihood: `P(keyword | Positive)`
  - Marginal Probability: `P(keyword)`
  - Posterior Probability: `P(Positive | keyword)`
- Displays the computed probabilities for each keyword using Bayes' Theorem.

## Requirements

- Python 3
- Jupyter Notebook or Google Colab
- Standard Python libraries only (no machine learning libraries)

## Output

Running the notebook prints the probability calculations for each selected keyword, demonstrating the application of Bayes' Theorem for sentiment analysis.

# Part 3: Gradient Descent Manual Calculation

## Overview

This document walks through **4 manual iterations of Gradient Descent** applied to a simple two-feature linear regression model. All calculations are performed using **matrix multiplication** rather than treating values as scalars. Each iteration was completed by one group member.

---

## Model & Setup

The linear model takes the form:

```
ŷ = X·m + b
```

where **X** is the feature matrix, **m** is the weight vector, and **b** is the bias vector added per sample.

### Given Values

| Parameter | Value |
|-----------|-------|
| Initial weights **m** | `[-1, 2]` |
| Initial bias **b** | `[1, 1]` |
| Feature matrix **X** | `[[1, 3], [4, 10]]` |
| Target values **y** | `[5, 6]` |
| Learning rate **α** | `0.01` |
| Number of iterations | `4` (one per group member) |

---

## Step 1 — Initial Predictions

Using matrix multiplication `X·m + b`, the initial predicted values are:

- **Row 1:** (1)(−1) + (3)(2) + 1 = **6**
- **Row 2:** (4)(−1) + (10)(2) + 1 = **17**

So `ŷ = [6, 17]`, giving initial residuals `e = ŷ − y = [1, 11]`.

---

## Step 2 — MSE Gradient Derivation

The Mean Squared Error cost function is:

```
J(m, b) = (1/n) · Σ (ŷᵢ − yᵢ)²
```

Applying the chain rule through `ŷ = X·m + b`:

```
∂J/∂m = Xᵀ · (ŷ − y)
∂J/∂b = (ŷ − y)
```

With `n = 2`, the scaling factor `2/n = 1`, so the gradient simplifies directly to the residual vector **e**.

---

## Step 3 — Gradient Descent Updates

The update rule applied at each iteration:

```
m ← m − α · (∂J/∂m)
b ← b − α · (∂J/∂b)
```

### Iteration 1 — Member 1

| Quantity | Value |
|----------|-------|
| ŷ | `[6, 17]` |
| e = ŷ − y | `[1, 11]` |
| ∂J/∂m | `[45, 113]` |
| ∂J/∂b | `[1, 11]` |
| **m (updated)** | `[-1.45, 0.87]` |
| **b (updated)** | `[0.99, 0.89]` |
| Cost J | 61.00 |

---

### Iteration 2 — Member 2

| Quantity | Value |
|----------|-------|
| ŷ | `[2.15, 3.79]` |
| e = ŷ − y | `[-2.85, -2.21]` |
| ∂J/∂m | `[-11.69, -30.65]` |
| ∂J/∂b | `[-2.85, -2.21]` |
| **m (updated)** | `[-1.3331, 1.1765]` |
| **b (updated)** | `[1.0185, 0.9121]` |
| Cost J | 6.50 |

---

### Iteration 3 — Member 3

| Quantity | Value |
|----------|-------|
| ŷ | `[3.2149, 7.3447]` |
| e = ŷ − y | `[-1.7851, 1.3447]` |
| ∂J/∂m | `[3.5937, 8.0917]` |
| ∂J/∂b | `[-1.7851, 1.3447]` |
| **m (updated)** | `[-1.3690, 1.0956]` |
| **b (updated)** | `[1.0364, 0.8987]` |
| Cost J | 2.50 |

---

### Iteration 4 — Member 4

| Quantity | Value |
|----------|-------|
| ŷ | `[2.9542, 6.3787]` |
| e = ŷ − y | `[-2.0458, 0.3787]` |
| ∂J/∂m | `[-0.5310, -2.2504]` |
| ∂J/∂b | `[-2.0458, 0.3787]` |
| **m (updated)** | `[-1.3637, 1.1181]` |
| **b (updated)** | `[1.0569, 0.8949]` |
| Cost J | 2.17 |

---

## Summary of All Updates

| Iteration | m (after) | b (after) | Cost J |
|-----------|-----------|-----------|--------|
| 0 (initial) | `[-1, 2]` | `[1, 1]` | 61.00 |
| 1 | `[-1.45, 0.87]` | `[0.99, 0.89]` | 61.00 |
| 2 | `[-1.3331, 1.1765]` | `[1.0185, 0.9121]` | 6.50 |
| 3 | `[-1.3690, 1.0956]` | `[1.0364, 0.8987]` | 2.50 |
| 4 | `[-1.3637, 1.1181]` | `[1.0569, 0.8949]` | 2.17 |

---

## Observations & Trend Analysis

### Cost J is consistently decreasing

The cost drops from **61.00 → 6.50 → 2.50 → 2.17** across the four iterations — a reduction of over **96%** from the initial value. This monotonic decrease confirms that the gradient descent updates are moving the parameters in the correct direction along the error surface.

### The large initial error is rapidly corrected

The starting residuals were highly imbalanced: `e = [1, 11]`. The second data point was dramatically over-predicted. By iteration 4, the second residual has collapsed from **11 to just 0.38**, while the first residual has stabilised around −2.0. The algorithm prioritised correcting the dominant source of error first.

### Parameters are converging

- **m₂** (weight on x₂) shifted from 2.0 → ~1.12, correcting for the model's over-reliance on the second feature.
- **m₁** (weight on x₁) settled around −1.36 after oscillating slightly near the initial value of −1.
- **b** values adjusted in small steps as the bias corrections became finer with each iteration.

The increasingly small parameter updates in iterations 3 and 4 (compared to the large jump in iteration 1) suggest the model is approaching a local minimum and would continue to converge with more iterations.

### Conclusion

Yes — **m and b are moving in a direction that reduces the error**. The gradient descent algorithm is working as expected: each update subtracts a fraction of the gradient from the current parameters, nudging them toward values that produce predictions closer to the true targets.


# Part 4: Gradient Descent in Code

Converts the Part 3 manual gradient descent calculations into a runnable Python script, with a SciPy-based numerical check on the gradient and two Matplotlib plots.

## What this does

Given a simple linear model:

```
y_hat = m1*x1 + m2*x2 + b
```

the script starts from the same initial values used in Part 3's hand calculations, runs gradient descent for a set number of iterations, and prints every step of the math (predictions, errors, gradients, parameter updates, cost) so it can be checked line-by-line against the manual work.

**Data (from Part 3):**

| | Value |
|---|---|
| `x` | `((1, 3), (4, 10))` |
| `y` | `(5, 6)` |
| initial `m` | `[-1, 2]` |
| initial `b` | `[1, 1]` |
| learning rate | `0.01` |
| iterations | `4` (one per group member) |

## Requirements

```
numpy
scipy
matplotlib
```

Install with:

```
pip install numpy scipy matplotlib
```

## Running it

```
python gradient_descent.py
```

This prints the full step-by-step trace to the console and saves two plots in the working directory.

## What it outputs

**Console:**
- Initial state (predictions, cost)
- Per-iteration: predictions, errors, analytical gradient, SciPy-verified gradient, updated `m`/`b`, updated cost
- Final `m`, `b`, predictions vs. actual `y`, final cost, and whether the cost trend is decreasing

**Files:**
- `params_over_iterations.png` — how `m1`, `m2`, and `b` change across iterations
- `cost_over_iterations.png` — how the MSE cost drops across iterations

## How it's structured

| Function | Purpose |
|---|---|
| `predict(X, m, b)` | Computes `y_hat` row by row (explicit, not vectorized in one line) |
| `mse(preds, y)` | Mean Squared Error — single shared definition of cost, used everywhere |
| `cost_flat(params, X, y)` | Same cost function repacked into one flat vector, needed for SciPy's numerical differentiation |
| `scipy_gradient(m, b, X, y)` | Uses `scipy.optimize.approx_fprime` to numerically estimate the gradient, as a check against the hand-derived formula |
| `analytical_gradient(X, y, m, b)` | The by-hand gradient formula (`dJ/dm`, `dJ/db`), computed with explicit loops |
| `run_gradient_descent(...)` | The main update loop; prints every step |
| `save_line_plot(...)` | One reusable plotting helper used for both charts |
| `main()` | Runs everything and saves the plots |

