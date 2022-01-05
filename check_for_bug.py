import numpy as np
import random
from numpy.lib.scimath import sqrt as csqr
import matplotlib.pyplot as plt
from tqdm import tqdm

# generate a random symmetric matrix
A = np.random.random((2000, 2000))
A = (A + A.T) / 2

L, V = np.linalg.eig(A)
L = csqr(L)
# L = np.expand_dims(L, axis=1)
L = np.diag(L)

s = 100
n = len(A)
list_of_available_indices = range(n)

runs = 1
results = []
im_vals = np.zeros((runs, A.shape[0]))
for i in tqdm(range(runs)):
	# choose samples
	sample_indices = np.sort(random.sample(list_of_available_indices, s))

	# compute B
	B = V @ L
	STB = B[sample_indices]

	alpha = 1#2*np.random.random()-1
	beta = -1#2*np.random.random()-1

	C = alpha * (B.T @ B) + beta * (STB.T @ STB)
	eigvals, eigvecs = np.linalg.eig(C)
	
	# print(im_vals[i,:].shape, eigvals.imag.shape)	
	im_vals[i, :] = eigvals.imag

	if all(np.isreal(eigvals)):
		results.append(1)
	else:
		results.append(0)

	

plt.hist(results)
# plt.show()
plt.savefig("figures/bug_check/bug_hist.pdf")

plt.gcf().clear()
#plt.imshow(im_vals)
#plt.colorbar()
plt.plot(im_vals[0,:])
plt.savefig("figures/bug_check/imag_vals.pdf")
