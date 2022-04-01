
"""
main code and entry point for all experiments
"""

import numpy as np
import random
from tqdm import tqdm
from src.display_codes import display, display_precomputed_error
from src.get_dataset import get_data
from src.similarities import hyperbolic_tangent, thin_plane_spline
import pickle
from src.sampler import sample_eig, sample_eig_default

"""
future speed improvement:
use scipy.sparse.linalg.eigs to compute k largest and smallest eigenvalues
"""
###########################################PARAMETERS############################################
# parameters
trials = 50
similarity_measure = "tps"#"tps", "ht" for kong, "default" for binary and random_sparse
search_rank = [0,1,2,3,-4,-3,-2,-1]
dataset_name = "kong"
#"multi_block_outer" 
#"multi_block_synthetic", "synthetic_tester", "binary", 
#"kong", "asymmetric", "facebook", "arxiv", "block", "synthetic_tester"
min_samples = 50
if dataset_name == "arxiv":
    max_samples = 5000
else:
    max_samples = 1000
# if dataset_name == "synthetic_tester":
#     max_samples = 500
# uncomment for run saved instance
# dataset_size = 5000
# dataset_name = "kong" #"kong", "facebook", "arxiv", "block", "erdos"
#################################################################################################

############################################# GRAB THE MATRICES #################################
if dataset_name == "kong":
    xy, dataset_size, min_samples, max_samples = get_data(dataset_name)
    if similarity_measure == "ht":
        similarity = hyperbolic_tangent
    if similarity_measure == "tps":
        similarity = thin_plane_spline
    true_mat = similarity(xy, xy)

if dataset_name != "kong":
    true_mat, dataset_size, min_samples, max_samples = get_data(dataset_name)
    print(true_mat.shape)

# uncommment when running the full code
true_spectrum = np.real(np.linalg.eigvals(true_mat))

# uncomment when running from saved values
# true_spectrum = np.zeros(len(true_mat))

print("loaded dataset")
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

# print(true_spectrum)

# comment out if you dont want to rerun and use only pickles
for i in tqdm(range(min_samples, max_samples, 10)):
    eig_vals = []
    error_vals = []
    for j in range(trials):
        # get eigenvalue
        if dataset_name == "kong":
            min_eig_single_round = sample_eig(xy, i, similarity, True, \
                                      rankcheck=search_rank)
        if dataset_name !="kong":
            min_eig_single_round = sample_eig_default(true_mat, i, True, \
                                                rankcheck=search_rank)
        # get error this round
        # error_single_round = np.log((min_eig_single_round - chosen_eig)**2)
        # uncomment following line for relative error
        # error_single_round = np.log(np.abs(min_eig_single_round - chosen_eig) / np.abs(chosen_eig))
        # error_single_round = np.log(np.abs(min_eig_single_round - chosen_eig) / dataset_size)
        error_single_round = np.abs(min_eig_single_round - chosen_eig) / float(dataset_size)
        # add to the local list
        eig_vals.append(min_eig_single_round)
        error_vals.append(error_single_round)
        

    # compute statistics from the local lists
    # print(eigvals.shape)
    mean_min_eig = np.mean(eig_vals, 0)
    std_min_eig = np.std(eig_vals, 0)
    mean_error = np.mean(error_vals, 0)
    std_error = np.std(error_vals, 0)
    percentile1 = np.percentile(error_vals, 20, axis=0)
    percentile2 = np.percentile(error_vals, 80, axis=0)

    # add statistics to the global list
    sample_eigenvalues_scaled.append(mean_min_eig)
    sample_eigenvalues_scaled_std.append(std_min_eig)
    tracked_errors.append(mean_error)
    tracked_errors_std.append(std_error)
    tracked_percentile1.append(percentile1)
    tracked_percentile2.append(percentile2)

############################# PREPARE TO SAVE OR DISPLAY ################################################

# comment out if you dont want to rerun and use only pickles
# convert to arrays
sample_eigenvalues_scaled = np.array(sample_eigenvalues_scaled)
sample_eigenvalues_scaled_std = np.array(sample_eigenvalues_scaled_std)
tracked_errors = np.array(tracked_errors)
tracked_errors_std = np.array(tracked_errors_std)
tracked_percentile1 = np.array(tracked_percentile1)
tracked_percentile2 = np.array(tracked_percentile2)

# comment out if you dont want to rerun and use only pickles
with open("pickle_files/new_pickles_"+dataset_name+"_"+similarity_measure+".pkl", "wb") as pickle_file:
    pickle.dump([sample_eigenvalues_scaled, sample_eigenvalues_scaled_std, \
            tracked_errors, tracked_errors_std, tracked_percentile1, tracked_percentile2], pickle_file)

# # uncomment to load from pickle file only
# with open("pickle_files/new_pickles_"+dataset_name+"_"+similarity_measure+".pkl", "rb") as pickle_file:
#     A = pickle.load (pickle_file)
# [sample_eigenvalues_scaled, sample_eigenvalues_scaled_std, \
#                   tracked_errors, tracked_errors_std, \
#                   tracked_percentile1, tracked_percentile2] = A

# # uncomment to load eigvalues from saved file
# f = open("figures/"+dataset_name+"/eigvals.txt", "r")
# if dataset_name == "kong":
#     f = open("figures/"+dataset_name+"/"+similarity_measure+"_eigvals.txt", "r")
# else:
#     f = open("figures/"+dataset_name+"/eigvals.txt", "r")
# all_lines = f.readlines()
# f.close()
# all_lines = [x.strip("\r\b") for x in all_lines]
# for i in range(len(all_lines)):
#     val_, index_ = all_lines[i].split()
#     val_ = float(val_)
#     index_ = int(index_)
#     true_spectrum[index_] = val_
# chosen_eig = true_spectrum[search_rank]


################################################################################################

###################################DISPLAY SECTION######################################


for i in range(len(search_rank)):
    display(dataset_name, similarity_measure, true_spectrum, dataset_size, search_rank[i], \
        sample_eigenvalues_scaled[:,i], sample_eigenvalues_scaled_std[:,i], max_samples, min_samples)
    display_precomputed_error(dataset_name, similarity_measure, tracked_errors[:,i], \
        dataset_size, search_rank[i], max_samples, \
        error_std=tracked_errors_std[:,i],\
        percentile1=tracked_percentile1[:,i], \
        percentile2=tracked_percentile2[:,i], min_samples=min_samples, \
        true_eigval=true_spectrum[search_rank[i]])
    # display_precomputed_error(dataset_name, similarity_measure, tracked_errors[:,i], \
    #     dataset_size, search_rank[i], max_samples, \
    #     error_std=tracked_errors_std[:,i],\
    #     percentile1=tracked_percentile1[:,i], \
    #     percentile2=tracked_percentile2[:,i], log = False)

