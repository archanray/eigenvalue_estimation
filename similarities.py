import numpy as np
from sklearn.metrics import pairwise_distances as euclid

def sigmoid(data1, data2, sigma=1):
    """
    sigmoid = tanh(xy/sigma+1.0)
    """
    similarity_matrix = np.matrix(data1) * np.matrix(data2.T)
    similarity_matrix = (similarity_matrix / sigma) + 1.0
    similarity_matrix = np.tanh(similarity_matrix)
    return similarity_matrix


def tps(data1, data2, sigma=1.0):
    """
    Ds = |x-y|^2 / sigma^2
    tps = Ds * ln(Ds)
    """
    eps=1e-16
    similarity_matrix = np.square(euclid(data1, data2))+eps
    similarity_matrix = similarity_matrix / (sigma**2)
    similarity_matrix = similarity_matrix * np.log(similarity_matrix)
    return similarity_matrix
