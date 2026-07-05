"""
Part 4: Gradient Descent in Code
---------------------------------
Converts the Part 3 manual gradient descent calculations into Python.

Model:      y_hat = m1*x1 + m2*x2 + b        (matrix form: y_hat = X @ m + b)
Cost:       J(m, b) = (1/n) * sum((y_hat - y)^2)      [Mean Squared Error]

Data (from Part 3):
    x = ((1, 3), (4, 10))
    y = (5, 6)
    initial m = [-1, 2]
    initial b = [1, 1]
"""

import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Data setup (matrix form, no scalars)
# ---------------------------------------------------------------------
X = np.array([[1, 3],
              [4, 10]], dtype=float)          # shape (n_samples=2, n_features=2)
y = np.array([5, 6], dtype=float)              # shape (2,)

m = np.array([-1.0, 2.0])                      # initial slope vector [m1, m2]
b = np.array([1.0, 1.0])                       # initial bias, one value per sample

learning_rate = 0.01
n_iterations = 4   # <-- set this to the number of members in your group

n_samples = X.shape[0]

# ---------------------------------------------------------------------
# 2. Prediction function
# ---------------------------------------------------------------------
def predict(X, m, b):
    """y_hat = X @ m + b, computed row by row so every step is visible."""
    preds = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        preds[i] = X[i, 0] * m[0] + X[i, 1] * m[1] + b[i]
    return preds


def mse(preds, y):
    """Mean Squared Error — single source of truth, used everywhere cost is needed."""
    return np.mean((preds - y) ** 2)

# ---------------------------------------------------------------------
# 3. Cost function J(m, b) = MSE
#    SciPy needs a single flat vector of parameters, so we pack
#    [m1, m2, b1, b2] into one array and unpack inside the function.
# ---------------------------------------------------------------------
def cost_flat(params, X, y):
    m_ = params[:2]
    b_ = params[2:]
    preds = predict(X, m_, b_)
    return mse(preds, y)

# ---------------------------------------------------------------------
# 4. Use SciPy to compute the derivative (numerical gradient) of J,
#    as required by the assignment. This is our "does my hand-derived
#    formula match reality" check.
# ---------------------------------------------------------------------
def scipy_gradient(m, b, X, y):
    params = np.concatenate([m, b])
    eps = np.sqrt(np.finfo(float).eps)
    grad = approx_fprime(params, cost_flat, eps, X, y)
    return grad[:2], grad[2:]   # dJ/dm , dJ/db

# ---------------------------------------------------------------------
# 5. Analytical gradient (the formula you derive by hand in Part 3)
#    dJ/dm_j = (2/n) * sum_i( (y_hat_i - y_i) * x_ij )
#    dJ/db_i = (2/n) * (y_hat_i - y_i)
# ---------------------------------------------------------------------
def analytical_gradient(X, y, m, b):
    n = X.shape[0]                           # derive sample count locally, no global dependency
    preds = predict(X, m, b)
    errors = preds - y                      # shape (n_samples,)

    grad_m = np.zeros(2)
    for j in range(2):                       # loop over m1, m2
        total = 0.0
        for i in range(n):                   # loop over data points
            total += errors[i] * X[i, j]
        grad_m[j] = (2 / n) * total

    grad_b = (2 / n) * errors                # dJ/db_i = (2/n)*(y_hat_i - y_i)

    return grad_m, grad_b

# ---------------------------------------------------------------------
# 6. Single plotting helper — reused for both charts instead of
#    duplicating the fig/ax/labels/grid/save boilerplate twice.
# ---------------------------------------------------------------------
def save_line_plot(x, series, xlabel, ylabel, title, filename, markers=None, colors=None):
    """series: dict[label] = y-values array. One shared plotting routine for any chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, (label, values) in enumerate(series.items()):
        marker = markers[idx] if markers else "o"
        color = colors[idx] if colors else None
        ax.plot(x, values, marker=marker, label=label, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# 7. Gradient descent loop — every math step printed, nothing hidden.
#    Cost is always computed via mse(), never re-derived inline.
# ---------------------------------------------------------------------
def run_gradient_descent(X, y, m, b, learning_rate, n_iterations):
    history = {"m": [m.copy()], "b": [b.copy()], "cost": []}

    print("=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    preds0 = predict(X, m, b)
    cost0 = mse(preds0, y)
    history["cost"].append(cost0)
    print(f"m = {m}, b = {b}")
    print(f"predictions y_hat = {preds0}")
    print(f"cost J = {cost0:.6f}")

    for it in range(1, n_iterations + 1):
        print("\n" + "=" * 70)
        print(f"ITERATION {it}")
        print("=" * 70)

        preds = predict(X, m, b)
        errors = preds - y
        print(f"y_hat        = {preds}")
        print(f"errors (y_hat - y) = {errors}")

        grad_m, grad_b = analytical_gradient(X, y, m, b)
        scipy_grad_m, scipy_grad_b = scipy_gradient(m, b, X, y)

        print(f"analytical dJ/dm = {grad_m}   |  scipy dJ/dm = {scipy_grad_m}")
        print(f"analytical dJ/db = {grad_b}   |  scipy dJ/db = {scipy_grad_b}")

        m_new = m - learning_rate * grad_m
        b_new = b - learning_rate * grad_b

        print(f"m_new = m - lr*grad_m = {m} - {learning_rate}*{grad_m} = {m_new}")
        print(f"b_new = b - lr*grad_b = {b} - {learning_rate}*{grad_b} = {b_new}")

        m, b = m_new, b_new

        new_cost = mse(predict(X, m, b), y)
        history["m"].append(m.copy())
        history["b"].append(b.copy())
        history["cost"].append(new_cost)

        print(f"updated cost J = {new_cost:.6f}")

    return m, b, history


def main():
    final_m, final_b, history = run_gradient_descent(X, y, m, b, learning_rate, n_iterations)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    final_preds = predict(X, final_m, final_b)
    print(f"final m = {final_m}")
    print(f"final b = {final_b}")
    print(f"final predictions y_hat = {final_preds}")
    print(f"actual y                = {y}")
    print(f"final cost J = {history['cost'][-1]:.6f}")

    trend = ("decreasing (moving toward lower error)"
              if history["cost"][-1] < history["cost"][0] else "not decreasing")
    print(f"\nCost trend across iterations: {trend}")

    m_hist = np.array(history["m"])
    b_hist = np.array(history["b"])
    cost_hist = np.array(history["cost"])
    iters = np.arange(len(cost_hist))

    save_line_plot(
        iters,
        {"m1": m_hist[:, 0], "m2": m_hist[:, 1], "b (per-sample value)": b_hist[:, 0]},
        xlabel="Iteration", ylabel="Parameter value",
        title="Parameters m and b over gradient descent iterations",
        filename="params_over_iterations.png",
        markers=["o", "o", "s"],
    )

    save_line_plot(
        iters,
        {"MSE cost": cost_hist},
        xlabel="Iteration", ylabel="MSE cost J(m, b)",
        title="Cost (error) over gradient descent iterations",
        filename="cost_over_iterations.png",
        markers=["o"], colors=["firebrick"],
    )

    print("\nSaved plots: params_over_iterations.png, cost_over_iterations.png")


if __name__ == "__main__":
    main()