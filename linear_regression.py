"""
Linear Regression from Scratch with NumPy
=========================================
Linear regression trained with gradient descent, implemented from scratch
(no machine learning libraries), applied to Yemen's foreign investment (x)
and unemployment (y) data, 1990-2000.

Run:  python linear_regression.py
"""

import numpy as np

# Set to True to draw the data and the fitted line (requires matplotlib).
PLOT = False

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
x_train = np.array([-1.03, 1.92, 3.99, 4.15, 0.05, -1.7, 0.92, -2.02, -3.47, -4.02, -0.06])
y_train = np.array([8.18, 8.13, 8.34, 8.34, 8.98, 9.58, 10.22, 10.83, 11.46, 11.56, 11.72])
years = np.array([1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000])


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def model_function(x, w, b):
    """Prediction of the line: y = w * x + b"""
    return w * x + b


def cost_function(x, y, w, b):
    """Mean squared error cost: (1 / 2m) * sum((prediction - y) ** 2)"""
    m = len(x)
    cost = 0
    for i in range(m):
        prediction = model_function(x[i], w, b)
        error = prediction - y[i]
        cost = cost + error ** 2
    return cost / (2 * m)


def compute_gradient(x, y, w, b):
    """Return the gradient of the cost with respect to w and b."""
    m = len(x)
    dw = 0
    db = 0
    for i in range(m):
        prediction = model_function(x[i], w, b)
        error = prediction - y[i]
        dw = dw + error * x[i]
        db = db + error
    return dw / m, db / m


def gradient_descent(x, y, w, b, alpha, num_iters, threshold):
    """
    Update w and b step by step to reduce the cost.
    Stops early when the cost improves by less than `threshold`.
    """
    previous_cost = float("inf")
    current_cost = previous_cost
    for _ in range(num_iters):
        dw, db = compute_gradient(x, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db

        current_cost = cost_function(x, y, w, b)
        improvement = previous_cost - current_cost
        previous_cost = current_cost
        if improvement < threshold:
            break
    return w, b, current_cost


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def main():
    w, b, final_cost = gradient_descent(
        x_train, y_train,
        w=0, b=0,
        alpha=0.1,
        num_iters=1000,
        threshold=0.000000001,
    )
    print("w =", w)
    print("b =", b)
    print("final cost =", final_cost)

    # Check against NumPy's built-in least squares fit
    w_ref, b_ref = np.polyfit(x_train, y_train, 1)
    print("Mine:  ", w, b)
    print("Check: ", w_ref, b_ref)

    if PLOT:
        import matplotlib.pyplot as plt

        plt.scatter(x_train, y_train, label="Data")
        x_line = np.linspace(x_train.min(), x_train.max(), 100)
        plt.plot(x_line, w * x_line + b, color="red", label="Fitted line")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
