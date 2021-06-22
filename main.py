import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from display_codes import display, display_precomputed_error
from get_dataset import get_data
from similarities import sigmoid, tps

def sample_eig(data, s, similarity_measure, scale=False, rankcheck=0):
    """
    input: original matrix
    output: sample eigenvalue
    """
    n = len(data)
    list_of_available_indices = range(n)
    sample_indices = np.sort(random.sample(list_of_available_indices, s))
    subsample_matrix = similarity_measure(data[sample_indices,:], data[sample_indices,:])
    all_eig_val_estimates = np.real(np.linalg.eigvals(subsample_matrix))
    all_eig_val_estimates.sort()
  
    min_eig = all_eig_val_estimates[rankcheck]

    if scale == False:
        return min_eig
    else:
        return n*min_eig/float(s)

############################################# GRAB THE MATRICES #################################
true_mat_sigmoid = sigmoid(xy, xy)
true_mat_tps = tps(xy, xy)
true_sigmoid_spectrum = np.real(np.linalg.eigvals(true_mat_sigmoid))
true_tps_spectrum = np.real(np.linalg.eigvals(true_mat_tps))
#################################################################################################

################################### COMPUTE ERRORS AND EIGS #####################################
trials = 50
similarity_measure = "sigmoid"
search_rank = 1
tracked_errors = []

if similarity_measure == "sigmoid":
    similarity = sigmoid
    true_spectrum = true_sigmoid_spectrum
if similarity_measure == "tps":
    similarity = tps
    true_spectrum = true_tps_spectrum

true_spectrum.sort()
chosen_eig = true_spectrum[search_rank]

sample_eigenvalues_scaled = []
for i in tqdm(range(10, 1000, 10)):
    trials_vals = 0
    error_total_round = 0
    for j in range(trials):
        min_eig_single_round = sample_eig(xy, i, similarity, True, \
                                      rankcheck=search_rank)
        error_total_round += np.log((min_eig_single_round - chosen_eig)**2)
        trials_vals += min_eig_single_round

    avg_min_eig = trials_vals / trials
    avg_error = error_total_round / trials
    sample_eigenvalues_scaled.append(avg_min_eig)
    tracked_errors.append(avg_error)
#################################################################################################


display("sigmoid", true_sigmoid_spectrum, dataset_size, search_rank, sample_eigenvalues_scaled)
display_precomputed_error("sigmoid", tracked_errors, dataset_size, search_rank, sample_eigenvalues_scaled)