import numpy as np
from tqdm import tqdm

L = list((0.1*2500)*np.ones(100))
L = np.array([1250.0] + L)
mask = np.random.random(101) < 0.5
L = np.where(mask, -1*L, L)
L = np.diag(L)
bound = 5.4

for i in tqdm(range(100)):
	V = np.random.random((2500,101))
	num = np.random.random()
	mask = np.random.random((2500,101)) < num
	V = np.where(mask, 0*V, V)
	I = (0.1*250.0)*np.ones(2500)
	V[:,0] = I

	Q, R= np.linalg.qr(V)

	A = Q @ L @ Q.T

	if np.min(A) >= -1*bound and np.max(A) <= 1*bound:
		print(num)