import numpy as np
from kernels import RBFKernel
from gpr import GaussianProcessRegression


def func(x):
    return np.sin(2 * np.pi * x) + np.cos(1.7 * 2 * np.pi * x)

def create_toy_data(func, low=0, high=1.0, n=10, std=1.0):
    x = np.random.uniform(low, high, n)
    y = func(x) + np.random.normal(scale=std, size=n)
    return x, y


# train:100, test:10
x_train, y_train = create_toy_data(func, low=0, high=1, n=100, std=0.1)
x_test,  y_test  = create_toy_data(func, low=0, high=1, n=10,  std=0.1)

print("x_train.shape :", x_train.shape)
print("y_train.shape :", y_train.shape)
print("x_test.shape  :", x_test.shape)
print("y_test.shape  :", y_test.shape)

kernel = RBFKernel(sigma=0.5)
model = GaussianProcessRegression(kernel, x_train, y_train, evar=1.0)
y_test_pred, y_test_pred_var= model.predict(x_test)

print("y_test             :", y_test)
print("y_test_pred (mean) :", y_test_pred)
