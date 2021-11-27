import numpy as np
import random
import matplotlib.pyplot as plt
from random import sample

n = 15000
eps = 0.1

A = np.ones((n,n))
num_blocks = round(1/(eps**2))
sample_block_sizes = round((eps**2)*n)

R = np.zeros_like(A)
Z = [-1, 1]

sample_block = np.ones((sample_block_sizes, sample_block_sizes))

block_start_row = []
block_end_row = []
block_start_col = []
block_end_col = []
start_row = 0
start_col = 0
for i in range(num_blocks):
	block_start_row.append(start_row)
	block_start_col.append(start_col)
	start_row += sample_block_sizes
	start_col += sample_block_sizes
	block_end_row.append(start_row)
	block_end_col.append(start_col)


row_id = 0
for i in range(num_blocks):
	col_id = 0

	for j in range(num_blocks):
		q = int(np.unique(R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]])[-1])

		if q == 0:
			flag = sample(Z,1)[-1]
			R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]] \
						= int(flag)*sample_block

			R[block_start_row[j]:block_end_row[j], block_start_col[i]:block_end_col[i]] \
						= flag*sample_block


plt.imshow(R)
plt.colorbar()
plt.savefig("figures/test_matrix_R.pdf")
plt.clf()
print("generated the matrix, proceeding to compute eigenvalues")

A = A+R
eigvals, eigvecs = np.linalg.eig(A)
E = np.abs(eigvals)
id_ = np.argmax(E)
max_eigval = np.real(eigvals[id_])
plt.scatter(range(n), eigvals)
plt.xlabel("indices")
plt.ylabel("eigenvalues")
plt.title("max abs val eig:"+str(max_eigval))
plt.savefig("figures/plot_of_eigenvalues.pdf")