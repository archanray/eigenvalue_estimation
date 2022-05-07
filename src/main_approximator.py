import numpy as np
from src.sampler import sample_eig_default
from tqdm import tqdm

def approximator(sampling_modes, min_samples, max_samples, trials, \
                true_mat, search_rank, chosen_eig, step=10):
    # details
    dataset_size = len(true_mat)
    # create loggers
    tracked_errors = {}
    tracked_errors_std = {}
    tracked_percentile1 = {}
    tracked_percentile2 = {}
    nnz_sm = {}
    
    # compute prob values for specific algorithms
    if "uniform random sample" in sampling_modes:
        unorm = np.ones(len(true_mat)) / len(true_mat)
    if "row norm sample" in sampling_modes:
        norm = np.linalg.norm(true_mat, axis=1)**2 / np.linalg.norm(true_mat)**2
    if "row nnz sample" in sampling_modes or any(i for i in sampling_modes if 'sparsity sampler' in i):
        nnz = np.count_nonzero(true_mat, axis=1, keepdims=False) / \
                                np.count_nonzero(true_mat, keepdims=False)
    #if "sparsity sampler" in sampling_modes:
    #    nnzA = np.count_nonzero(true_mat)
    if any(i for i in sampling_modes if 'sparsity sampler' in i):
        nnzA = np.count_nonzero(true_mat)

    print("nnzA:", nnzA)

    # create more loggers
    for m in sampling_modes:
        tracked_errors[m] = []
        tracked_errors_std[m] = []
        tracked_percentile1[m] = []
        tracked_percentile2[m] = []
        if "sparsity sampler" in m:
            nnz_sm[m] = []
    # Analysis block (not needed for code): plot row norms
    # disply_prob_histogram(norm, dataset_name)

    # finally run the trials for multiple iterations
    for i in tqdm(range(min_samples, max_samples, 10)):
        eig_vals = {}
        error_vals = {}
        if "sparsity sampler" in m:
            nnz_submatrix = {}
        for m in sampling_modes:
            eig_vals[m] = []
            error_vals[m] = []
            if "sparsity sampler" in m:
                nnz_submatrix[m] = []
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
                if "sparsity sampler" in m:
                    # split name and parameters to get the multiplier
                    mult = float(m.split("_")[1])
                    min_eig_single_round, nnz_subsample_matrtix = \
                                           sample_eig_default(true_mat, i, scale=False, \
                                                              rankcheck=search_rank,
                                                              norm=nnz, nnzA=nnzA, method=m, \
                                                              multiplier=mult)
                    # min_eig_single_round = op[0:-1]
                    # nnz_subsample_matrtix = op[-1]
                # get error this round
                if "uniform random sample" in m and len(m) == 1:
                    error_single_round = np.abs(min_eig_single_round - chosen_eig) / \
                                    float(n)
                else:
                    error_single_round = np.abs(min_eig_single_round - chosen_eig) / \
                                    float(np.sqrt(nnzA))

                                    # float(dataset_size)
                # add to the local list
                eig_vals[m].append(min_eig_single_round)
                error_vals[m].append(error_single_round)

                if "sparsity sampler" in m:
                    nnz_submatrix[m].append(nnz_subsample_matrtix)
        
        for m in sampling_modes:
            mean_error = np.mean(error_vals[m], 0)
            percentile1 = np.percentile(error_vals[m], 20, axis=0)
            percentile2 = np.percentile(error_vals[m], 80, axis=0)

            tracked_errors[m].append(mean_error)
            tracked_percentile1[m].append(percentile1)
            tracked_percentile2[m].append(percentile2)

            if "sparsity sampler" in m:
                nnz_sm[m].append(np.mean(nnz_submatrix[m], 0))

    return tracked_errors, tracked_percentile1, tracked_percentile2, nnz_sm
