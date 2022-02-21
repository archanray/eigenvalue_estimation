import numpy as np
import networkx as nx
import idx2numpy
from sklearn.preprocessing import normalize
from random import sample
import os
from tqdm import tqdm
import pickle
import random
import matplotlib.pyplot as plt
from sampler import sample_eig_default
from utils import get_distance
from display_codes import display_combined_error, disply_prob_histogram
from get_dataset import get_data
from similarities import hyperbolic_tangent, thin_plane_spline

# Parameters
trials = 50
search_rank = [0,1,2,3,-4,-3,-2,-1]
dataset_name = "arxiv"
# dataset_name = "erdos", "MNIST", "block", "facebook", "kong", "multi_block_outer", "arxiv"
name_adder = "norm_v_random_v_nnz"
# name_adder = "random"
sampling_modes = ["uniform random sample", "row norm sample", "row nnz sample"]

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

print("loaded dataset")
print("||A||_infty:", np.max(true_mat))

# set up output loggers
tracked_errors = {}
tracked_errors_std = {}
tracked_percentile1 = {}
tracked_percentile2 = {}
true_spectrum.sort()
chosen_eig = true_spectrum[search_rank]

# compute prob values for specific algorithms
unorm = np.ones(len(true_mat)) / len(true_mat)
norm = np.linalg.norm(true_mat, axis=1)**2 / np.linalg.norm(true_mat)**2
nnz = np.count_nonzero(true_mat, axis=1, keepdims=False) / \
        np.count_nonzero(true_mat, keepdims=False)

# create more loggers
for m in sampling_modes:
    tracked_errors[m] = []
    tracked_errors_std[m] = []
    tracked_percentile1[m] = []
    tracked_percentile2[m] = []

# Analysis block (not needed for code): plot row norms
# disply_prob_histogram(norm, dataset_name)

# finally run the trials for multiple iterations
for i in tqdm(range(min_samples, max_samples, 10)):
    eig_vals = {}
    error_vals = {}
    for m in sampling_modes:
        eig_vals[m] = []
        error_vals[m] = []
    for j in range(trials):
        # for each trial, run on every modes get its eigenvalue
        for m in sampling_modes:
            if m == "row norm sample":
                min_eig_single_round = sample_eig_default(true_mat, i, False, \
                                                          rankcheck=search_rank, \
                                                          norm=norm)
            if m == "row nnz sample":
                min_eig_single_round = sample_eig_default(true_mat, i, False, \
                                                          rankcheck=search_rank, \
                                                          norm=nnz)
            if m == "uniform random sample":
                min_eig_single_round = sample_eig_default(true_mat, i, False,
                                                          rankcheck=search_rank,
                                                          norm=unorm)
            # get error this round
            error_single_round = np.abs(min_eig_single_round - chosen_eig) / \
                                float(dataset_size)
            # add to the local list
            eig_vals[m].append(min_eig_single_round)
            error_vals[m].append(error_single_round)
    
    for m in sampling_modes:
        mean_error = np.mean(error_vals[m], 0)
        percentile1 = np.percentile(error_vals[m], 20, axis=0)
        percentile2 = np.percentile(error_vals[m], 80, axis=0)

        tracked_errors[m].append(mean_error)
        tracked_percentile1[m].append(percentile1)
        tracked_percentile2[m].append(percentile2)

# finally visualize the file
for m in sampling_modes:
    tracked_errors[m] = np.array(tracked_errors[m])
    tracked_percentile1[m] = np.array(tracked_percentile1[m])
    tracked_percentile2[m] = np.array(tracked_percentile2[m])

for i in range(len(search_rank)):
    tracked_errors_rank = {}
    percentile1_rank = {}
    percentile2_rank = {}

    for m in sampling_modes:
        tracked_errors_rank[m] = tracked_errors[m][:, i]
        percentile1_rank[m] = tracked_percentile1[m][:, i]
        percentile2_rank[m] = tracked_percentile2[m][:, i]
    display_combined_error(sampling_modes, dataset_name, tracked_errors_rank, \
        dataset_size, search_rank[i], max_samples, \
        percentile1=percentile1_rank, \
        percentile2=percentile2_rank, min_samples=min_samples, \
        true_eigval=true_spectrum[search_rank[i]], name_adder=name_adder)
