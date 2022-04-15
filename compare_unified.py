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
from src.viz import plot_all_errors
from src.utils import get_distance
from src.display_codes import disply_prob_histogram
from src.get_dataset import get_data
from src.similarities import hyperbolic_tangent, thin_plane_spline

# Parameters
trials = 50
search_rank = [0,1,2,3,-4,-3,-2,-1]
dataset_name = "facebook"
# dataset_name = "erdos", "MNIST", "block", "facebook", "kong", "multi_block_outer", "arxiv"
name_adder = "nnz_sparse_multi"
# name_adder = "random"
sampling_modes = ["row nnz sample", "sparsity sampler_1","sparsity sampler_100", "sparsity sampler_500", "sparsity sampler_1000", "sparsity sampler_2500", "sparsity sampler_5000", "sparsity sampler_10000", "sparsity sampler_25000", "sparsity sampler_50000"]

if dataset_name == "kong":
    similarity_measure = "tps" # "tps", "ht", 

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

print("loaded dataset")
print("||A||_infty:", np.max(true_mat))

# set up output loggers
true_spectrum.sort()
chosen_eig = true_spectrum[search_rank]
print("chosen eigs:", chosen_eig)

# compute the errors
tracked_errors, tracked_percentile1, tracked_percentile2 = approximator(\
            sampling_modes, min_samples, max_samples, trials, true_mat, search_rank, chosen_eig)

# visualize the errors
plot_all_errors(tracked_errors, tracked_percentile1, tracked_percentile2, \
            sampling_modes, dataset_name, dataset_size, \
            search_rank, max_samples, min_samples, \
            true_spectrum, name_adder)
