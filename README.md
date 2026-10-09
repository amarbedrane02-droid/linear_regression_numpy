# Linear Regression from Scratch with NumPy

Linear regression trained with **gradient descent**, implemented from scratch
(no machine learning libraries), applied to Yemen's foreign investment and
unemployment data (1990–2000).

## What it does
- `model_function`: the line `y = w * x + b`
- `cost_function`: mean squared error
- `compute_gradient`: computes the gradients for `w` and `b`
- `gradient_descent`: updates `w` and `b` until the cost stops improving
- Compares the result with NumPy's `np.polyfit` to verify it

## Result
| | w | b |
|---|---|---|
| This code | -0.37669 | 9.71460 |
| `np.polyfit` | -0.37669 | 9.71469 |

## How to run
```bash
pip install -r requirements.txt
python linear_regression.py
```
Set `PLOT = True` at the top of the file to also draw the fitted line (needs matplotlib).

## Files
- `linear_regression.py` — the full code
- `requirements.txt` — dependencies
