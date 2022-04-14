import numpy as np
from src.sampler import sample_eig_default
from tqdm import tqdm

def modify_matrix_for_sparsity(true_mat):
    true_mat = true_mat - np.diag(np.diag(true_mat))
    return true_mat

def approximator(sampling_modes, min_samples, max_samples, trials, \
                true_mat, search_rank, chosen_eig, step=10):
    # details
    dataset_size = len(true_mat)
    # create loggers
    tracked_errors = {}
    tracked_errors_std = {}
    tracked_percentile1 = {}
    tracked_percentile2 = {}

    # compute prob values for specific algorithms
    if "uniform random sample" in sampling_modes:
        unorm = np.ones(len(true_mat)) / len(true_mat)
    if "row norm sample" in sampling_modes:
        norm = np.linalg.norm(true_mat, axis=1)**2 / np.linalg.norm(true_mat)**2
    if "row nnz sample" in sampling_modes or "sparsity sampler" in sampling_modes:
        nnz = np.count_nonzero(true_mat, axis=1, keepdims=False) / \
                                np.count_nonzero(true_mat, keepdims=False)
    if "sparsity sampler" in sampling_modes:
        nnzA = np.count_nonzero(true_mat)

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
                    min_eig_single_round = sample_eig_default(true_mat, i, scale=False, \
                                                              rankcheck=search_rank, \
                                                              norm=norm, method=m)
                if m == "row nnz sample":
                    min_eig_single_round = sample_eig_default(true_mat, i, scale=False, \
                                                              rankcheck=search_rank, \
                                                              norm=nnz, method=m)
                if m == "uniform random sample":
                    min_eig_single_round = sample_eig_default(true_mat, i, scale=False,
                                                              rankcheck=search_rank,
                                                              norm=unorm, method=m)
                if m == "sparsity sampler":
                    min_eig_single_round = sample_eig_default(true_mat, i, scale=False,
                                                              rankcheck=search_rank,
                                                              norm=nnz, nnzA=nnzA, method=m, multiplier=500)
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

    return tracked_errors, tracked_percentile1, tracked_percentile2
