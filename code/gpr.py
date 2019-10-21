import numpy as np

class GaussianProcessRegression():
    """
    kernel: class
    x_train: train data
    y_train: train label
    evar: variance of epsilon_i
    """
    def __init__(self, kernel, x_train, y_train, evar=1.):
        self.kernel = kernel
        self.x_train = x_train
        self.y_train = y_train
        self.evar = evar
        self.K_xx, self.covariance, self.precision = self.fit_kernel(self.x_train)

    def fit_kernel(self, x):
        n = x.shape[0]
    	# Gram matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i][j] = self.kernel(x[i], x[j])
	    # Covariance matrix
        covariance = K + np.identity(n) * self.evar # shape=(n,n)
	    # Precision matrix
        precision = np.linalg.inv(covariance) # shape=(n,n)
        return K, covariance, precision
    
    def predict(self, x_test):
        K_xx, C_xx, P_xx = self.fit_kernel(self.x_train)
        K_tt, _,    _    = self.fit_kernel(x_test)
        
        if x_test.shape[0] == 1:
            x_test = np.squeeze(x_test)
            k_tx = np.squeeze(np.array([self.kernel(xi, x_test) for xi in self.x_train])) # k_test.shape=(n,)
            y_test_mean = np.dot(k_tx, np.dot(P_xx, self.y_train))
            y_test_var = self.kernel(x_test, x_test) - np.dot(k_tx, np.dot(P_xx, self.y_train))
        else:
            K_tx=[]
            for j in range(x_test.shape[0]):
                K_tx_j = np.array([self.kernel(xi, x_test[j]) for xi in self.x_train])
                K_tx.append(K_tx_j)
            K_tx = np.array(K_tx)
            y_test_mean = np.dot(K_tx, np.dot(P_xx, self.y_train))
            y_test_var  = K_tt - np.dot(K_tx, np.dot(P_xx, K_tx.T))
        return y_test_mean, y_test_var
