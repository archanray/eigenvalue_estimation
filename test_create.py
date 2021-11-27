import numpy as np
from tqdm import tqdm

n = 5000
eps = 0.1
L = list(eps*n*np.ones(100))
L = np.array([(n/2.0)] + L)
mask = np.random.random(101) < 0.5
L = np.where(mask, -1*L, L)
L = np.diag(L)
# bound = 5.4


V = np.random.random((n,101))
num = 0.4
mask = np.random.random((n,101)) < num
V = np.where(mask, 0*V, V)
I = (1/np.sqrt(n))*np.ones(n)
V[:,0] = I

Q, R= np.linalg.qr(V)

A = Q @ L @ Q.T
