"""
code to test different powers of epsilon
"""

import numpy as np
import random
from tqdm import tqdm
from display_codes import display, display_precomputed_error
from get_dataset import get_data
from similarities import hyperbolic_tangent, thin_plane_spline
import pickle

def sample_eig_default(data_matrix, s, scale=False, rankcheck=0):
    """
    input: original matrix
    output: sample eigenvalue
    requires data matrix to be fully instantiated
    """
    n = len(data_matrix)
    list_of_available_indices = range(n)
    sample_indices = np.sort(random.sample(list_of_available_indices, s))
    subsample_matrix = data_matrix[sample_indices][:, sample_indices]
    # useful for only hermitian matrices
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()

    min_eig = np.array(all_eig_val_estimates)[rankcheck]
    if scale == False:
        return min_eig
    else:
        return n*min_eig/float(s)

######################################################################################
# parameters to retrieve dataset
trials = 1#50
similarity_measure = "default"
search_rank = [0,1,2,3,-4,-3,-2,-1]
dataset_name = "multi_block_outer"
min_samples = 50
max_samples = 60#1000
steps = 10
#################################################################################################

################################### COMPUTE ERRORS AND EIGS #####################################
# logging data-structures
# set power of eps and loop around it
eps_pows = [1.5, 2, 2.5, 3, 3.5]
eps_means_per_round = []
eps_percent1_per_round = []
eps_percent2_per_round = []

# comment out if you dont want to rerun and use only pickles
for pows in eps_pows:
    print(pows)
    error_val_means = []
    percentile1 = []
    percentile2 = []
    for s in tqdm(range(min_samples, max_samples, steps)):
        # compute the eps for the given s: s=1/eps^pows ==> eps = 1/s^{1/pows}
        local_eps = 1 / (s ** (1/pows))
        
        # run experiment trials        
        error_vals = []
        for j in range(trials):
            # create the matrix 
            matrix, n, _, _ = get_data(dataset_name, eps=local_eps, plot_mat=False)

            # get the true spectrum and the subset we are guning for
            true_spectrum = np.real(np.linalg.eigvals(matrix))
            true_spectrum.sort()
            chosen_eig = true_spectrum[search_rank]

            # compute the approximate eigenvalues
            min_eig_single_round = sample_eig_default(matrix, s, True, rankcheck=search_rank)
            # compute the error this round
            error_single_round = np.abs(min_eig_single_round - chosen_eig) / float(n)
            error_vals.append(error_single_round)

        # save the values for this specific s
        error_val_means.append(np.mean(error_vals, 0))
        percentile1.append(np.percentile(error_vals, 20, axis=0))
        percentile2.append(np.percentile(error_vals, 80, axis=0))

    # bookkeeping
    eps_means_per_round.append(error_val_means)
    eps_percent1_per_round.append(percentile1)
    eps_percent2_per_round.append(percentile2)

# comment out if you dont want to rerun and use only pickles
with open("pickle_files/multi_eps_new_pickles_"+dataset_name+".pkl", "wb") as pickle_file:
    pickle.dump([eps_means_per_round, error_val_stds, min_samples, max_samples, steps, eps_pows], pickle_file)