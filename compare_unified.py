import numpy as np
import networkx as nx
import idx2numpy
from sklearn.preprocessing import normalize
from random import sample
import os
import pickle
import random
import matplotlib.pyplot as plt
from src.main_approximator import approximator
from src.viz import plot_all_errors, plot_all_nnz, plot_eigval_vs_nnzA
from src.utils import get_distance
from src.display_codes import disply_prob_histogram
from src.get_dataset import get_data
from src.similarities import hyperbolic_tangent, thin_plane_spline
from copy import copy

# Parameters
trials = 10
#search_rank = [0,1,2,3,-4,-3,-2,-1]
a = list(range(-20,0))
b = list(range(0,20))
search_rank = b+a
dataset_name = "facebook"
# dataset_name = "erdos", "MNIST", "block", "facebook", "kong", "multi_block_outer", "arxiv", "tridiagonal"
name_adder = "test"
# name_adder = "random"
# sampling modes options "row nnz sample", "uniform random sample", "sparsity sampler_0.1" can change the float here
sampling_modes = ["row nnz sample", "uniform random sample", "sparsity sampler_0.1"]

if dataset_name == "kong":
    similarity_measure = "ht" # "tps", "ht", 

# Get the dataset
if dataset_name == "kong":
    xy, dataset_size, min_samples, max_samples = get_data(dataset_name)
    if similarity_measure == "ht":
        similarity = hyperbolic_tangent
    if similarity_measure == "tps":
        similarity = thin_plane_spline
    true_mat = similarity(xy, xy)

if dataset_name != "kong":
    true_mat, dataset_size, min_samples, max_samples = get_data(dataset_name)
# print(true_mat.shape)

# Analysis block (not needed for code): select duplicate row and col
# unq, count = np.unique(true_mat, axis=0, return_counts=True)
# repeated_groups = unq[count > 1]
# unq = unq[count == 1]
# print(unq.shape)
# print(len(repeated_groups))

# uncommment when running the full code
true_spectrum = np.real(np.linalg.eigvals(true_mat))

# uncomment when running from saved values
# true_spectrum = np.zeros(len(true_mat))

if any(i for i in sampling_modes if "sparsity sampler" in i):
    plot_eigval_vs_nnzA(true_spectrum, np.count_nonzero(true_mat), dataset_name)

print("loaded dataset")
print("||A||_infty:", np.max(true_mat))

# set up output loggers
true_spectrum.sort()
chosen_eig = true_spectrum[search_rank]
print("chosen eigs:", chosen_eig)

# compute the errors
tracked_errors, tracked_percentile1, tracked_percentile2, nnz_sm = approximator(\
            sampling_modes, min_samples, max_samples, trials, true_mat, search_rank, chosen_eig)

# set up baseline
if any(i for i in sampling_modes if "sparsity sampler" in i):
    divisor = np.sqrt(np.count_nonzero(true_mat))
else:
    divisor = len(true_mat)

nom = "lambda_by_nnz"
sampling_modes.append(nom)
tracked_errors[nom] = []
tracked_percentile1[nom] = []
tracked_percentile2[nom] = []
len_samples = len(tracked_errors[sampling_modes[0]])
vals = np.array(abs(chosen_eig) / divisor)
vals = vals+1e-60 # adding this for stability in computing logarithm even when observing lambda_i = 0
for i in range(len_samples):
    tracked_errors[nom].append(vals)
    tracked_percentile1[nom].append(vals)
    tracked_percentile2[nom].append(vals)

with open("pickle_files/"+dataset_name+"_"+name_adder+".pkl", "wb") as f:
    pickle.dump([tracked_errors, tracked_percentile1, tracked_percentile2, sampling_modes, dataset_name, dataset_size, \
            search_rank, max_samples, min_samples, \
            true_spectrum, name_adder], f)

# visualize the errors
plot_all_errors(tracked_errors, tracked_percentile1, tracked_percentile2, \
            sampling_modes, dataset_name, dataset_size, \
            search_rank, max_samples, min_samples, \
            true_spectrum, name_adder)

# visualize non-zeros elements in the submatrix of the sparsity sampler
plot_all_nnz(nnz_sm, sampling_modes, dataset_name, min_samples, max_samples)
