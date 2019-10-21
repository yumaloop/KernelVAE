import numpy as np
import tensorflow as tf
from kernels import RBFKernel, NNArccosineKernel
from gpr import GaussianProcessRegression

def load_mnist():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((-1, 784,)).astype('float32') # / 255.
    X_test  = X_test.reshape((-1, 784,)).astype('float32')  # / 255.
    return (X_train, y_train), (X_test, y_test) 

def main():
    # train:100, test:10
    (x_train, y_train), (x_test, y_test) = load_mnist()

    x_train = x_train[:10]
    x_test  = x_test[11:12]
    y_train = y_train[:10]
    y_test  = y_test[11:12]

    print("x_train.shape :", x_train.shape)
    print("y_train.shape :", y_train.shape)
    print("x_test.shape  :", x_test.shape)
    print("y_test.shape  :", y_test.shape)

    # kernel = RBFKernel(sigma=0.5)
    kernel = NNArccosineKernel(wvar=2)

    model = GaussianProcessRegression(kernel, x_train, y_train, evar=1.0)
    y_test_pred, y_test_pred_var= model.predict(x_test)

    print("y_test             :", y_test)
    print("y_test_pred (mean) :", y_test_pred)

if __name__ == '__main__':
    main()
