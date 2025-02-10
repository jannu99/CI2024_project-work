import numpy as np
import s331391

problem = np.load('data/problem_2.npz')
x = problem['x']
y = problem['y']
print(x.shape)

y_pred_f = s331391.f2(x)

# MSE
mse_f = np.mean((y - y_pred_f) ** 2)
print(f"\nMSE: {mse_f:.6e}")
