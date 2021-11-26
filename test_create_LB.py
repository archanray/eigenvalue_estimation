import numpy as np
import random

n = 2500
eps = 0.1
A = np.ones((n,n))
num_blocks = int(1/(eps**4))
sample_block_sizes = int((eps**2)*n)
list_of_indices = list(range(n))

samples = []
for i in range(num_blocks):
	print(len(list_of_indices), sample_block_sizes, num_blocks)
	sampled_indices = random.sample(list_of_indices, k=sample_block_sizes)
	list_of_indices = list(set(list_of_indices) - set(sampled_indices))
	samples.append(sampled_indices)

print(len(samples))
