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
trials = 50
similarity_measure = "default"
search_rank = [0,1,2,3,-4,-3,-2,-1]
dataset_name = "multi_block_outer"
min_samples = 50
max_samples = 1000



true_mat, dataset_size, min_samples, max_samples = get_data(dataset_name, eps=local_eps, print_mat=False)
# uncommment when running the full code
true_spectrum = np.real(np.linalg.eigvals(true_mat))
# uncomment when running from saved values
# true_spectrum = np.zeros(len(true_mat))
print(">>>>>>>>>loaded dataset>>>>>>>>")
print("||A||_infty:", np.max(true_mat))
#################################################################################################

################################### COMPUTE ERRORS AND EIGS #####################################
# logging data-structures
sample_eigenvalues_scaled = []
sample_eigenvalues_scaled_std = []
tracked_errors = []
tracked_errors_std = []
tracked_percentile1 = []
tracked_percentile2 = []
true_spectrum.sort()
chosen_eig = true_spectrum[search_rank]

# comment out if you dont want to rerun and use only pickles
for i in tqdm(range(min_samples, max_samples, 10)):
    eig_vals = []
    error_vals = []
    for j in range(trials):
        # get eigenvalue
        error_single_round = np.abs(min_eig_single_round - chosen_eig) / float(dataset_size)