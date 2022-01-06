import numpy as np
import random
# from numpy.lib.scimath import sqrt as csqr
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import sqrtm

# generate a random symmetric matrix
#A = np.random.random((2000, 2000))
#A = (A + A.T) / 2
n = 1000
A = 2*np.random.randint(1,3, (n,n))-3
A = A - np.tril(A, -1) + np.triu(A,1).T

L, V = np.linalg.eig(A)
L = np.diag(L)
L_h = sqrtm(L)
#L_h = np.diag(L_h)
B = V @ L_h
BTB = L

s = 100
list_of_available_indices = range(n)

runs = 1
results = []
im_vals = np.zeros((runs, A.shape[0]))

for i in tqdm(range(runs)):
	# choose samples
	sample_indices = np.sort(random.sample(list_of_available_indices, s))

	# compute B_sample
	STB = B[sample_indices,:]

	alpha = 1000#2*np.random.random()-1
	beta = -500#2*np.random.random()-1

	C = beta * BTB + alpha * (n/s) * (STB.T @ STB)
	eigvals = np.linalg.eigvals(C)
	
	im_vals[i, :] = eigvals.imag

	if all(np.isreal(eigvals)):
		results.append(1)
	else:
		results.append(0)

plt.gcf().clear()
plt.hist(results)
# plt.show()
plt.savefig("figures/bug_check/bug_hist.pdf")

plt.gcf().clear()
plt.plot(im_vals[0,:])
plt.savefig("figures/bug_check/imag_vals.pdf")
