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

