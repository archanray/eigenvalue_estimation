import numpy as np

# A very fast distance function

def get_distance(X, Y):
    """
    fast euclidean distance computation:
    X shape is n x features
    """
    num_test = X.shape[0]
    num_train = Y.shape[0]
    dists = np.zeros((num_test, num_train))
    sum1 = np.sum(np.power(X,2), axis=1)
    sum2 = np.sum(np.power(Y,2), axis=1)
    sum3 = 2*np.dot(X, Y.T)
    dists = sum1.reshape(-1,1) + sum2
    dists = np.sqrt(dists - sum3)
    dists = dists / np.max(dists)
    return dists
