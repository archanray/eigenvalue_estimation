import numpy as np
import random
from tqdm import tqdm
from display_codes import display, display_precomputed_error
from get_dataset import get_data
from similarities import sigmoid, tps
import pickle

"""
future speed improvement:
use scipy.sparse.linalg.eigs to compute k largest and smallest eigenvalues
"""

def sample_eig(data, s, similarity_measure, scale=False, rankcheck=0):
    """
    input: original matrix
    output: sample eigenvalue
    """
    n = len(data)
    list_of_available_indices = range(n)
    sample_indices = np.sort(random.sample(list_of_available_indices, s))
    subsample_matrix = similarity_measure(data[sample_indices,:], data[sample_indices,:])
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()
  
    min_eig = np.array(all_eig_val_estimates)[rankcheck]

    if scale == False:
        return min_eig
    else:
        return n*min_eig/float(s)

def sample_eig_default(data_matrix, s, scale=False, rankcheck=0):
    """
    input: opriginal matrix
    output: sample eigenvalue
    """
    n = len(data_matrix)
    list_of_available_indices = range(n)
    sample_indices = np.sort(random.sample(list_of_available_indices, s))
    subsample_matrix = data_matrix[sample_indices][:, sample_indices]
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()

    min_eig = np.array(all_eig_val_estimates)[rankcheck]
    if scale == False:
        return min_eig
    else:
        return n*min_eig/float(s)

###########################################PARAMETERS############################################
# parameters
trials = 50
similarity_measure = "default" #"tps", "sigmoid" for kong, "default" for binary and random_sparse
search_rank = [0,1,2,3,-4,-3,-2,-1]
dataset_name = "arxiv" #"binary", "kong", "asymmetric", "facebook", "arxiv", "block"
min_samples = 230
if dataset_name == "arxiv":
    max_samples = 5000
else:
    max_samples = 1000
# uncomment for run saved instance
# dataset_size = 5000
#################################################################################################

############################################# GRAB THE MATRICES #################################
if dataset_name == "kong":
    xy, dataset_size = get_data(dataset_name)
    if similarity_measure == "sigmoid":
        similarity = sigmoid
    if similarity_measure == "tps":
        similarity = tps
    true_mat = similarity(xy, xy)

if dataset_name != "kong":
    true_mat, dataset_size = get_data(dataset_name)
    print(true_mat.shape)

# # uncommment when running the full code
# true_spectrum = np.real(np.linalg.eigvals(true_mat))
# uncomment when running from saved values
true_spectrum = np.zeros(len(true_mat))
print("loaded dataset")
print("||A||_infty:", np.max(true_mat))
#################################################################################################

################################### COMPUTE ERRORS AND EIGS #####################################
sample_eigenvalues_scaled = []
sample_eigenvalues_scaled_std = []
tracked_errors = []
tracked_errors_std = []
tracked_percentile1 = []
tracked_percentile2 = []
true_spectrum.sort()
chosen_eig = true_spectrum[search_rank]

# # comment out if you dont want to rerun and use only pickles
# for i in tqdm(range(min_samples, max_samples, 10)):
#     eig_vals = []
#     error_vals = []
#     for j in range(trials):
#         # get eigenvalue
#         if dataset_name == "kong":
#             min_eig_single_round = sample_eig(xy, i, similarity, True, \
#                                       rankcheck=search_rank)
#         if dataset_name !="kong":
#             min_eig_single_round = sample_eig_default(true_mat, i, True, \
#                                                 rankcheck=search_rank)
#         # get error this round
#         # error_single_round = np.log((min_eig_single_round - chosen_eig)**2)
#         # uncomment following line for relative error
#         # error_single_round = np.log(np.abs(min_eig_single_round - chosen_eig) / np.abs(chosen_eig))
#         # error_single_round = np.log(np.abs(min_eig_single_round - chosen_eig) / dataset_size)
#         error_single_round = np.abs(min_eig_single_round - chosen_eig) / float(dataset_size)
#         # add to the local list
#         eig_vals.append(min_eig_single_round)
#         error_vals.append(error_single_round)
        

#     # compute statistics from the local lists
#     # print(eigvals.shape)
#     mean_min_eig = np.mean(eig_vals, 0)
#     std_min_eig = np.std(eig_vals, 0)
#     mean_error = np.mean(error_vals, 0)
#     std_error = np.std(error_vals, 0)
#     percentile1 = np.percentile(error_vals, 20, axis=0)
#     percentile2 = np.percentile(error_vals, 80, axis=0)

#     # add statistics to the global list
#     sample_eigenvalues_scaled.append(mean_min_eig)
#     sample_eigenvalues_scaled_std.append(std_min_eig)
#     tracked_errors.append(mean_error)
#     tracked_errors_std.append(std_error)
#     tracked_percentile1.append(percentile1)
#     tracked_percentile2.append(percentile2)

# # convert to arrays
# sample_eigenvalues_scaled = np.array(sample_eigenvalues_scaled)
# sample_eigenvalues_scaled_std = np.array(sample_eigenvalues_scaled_std)
# tracked_errors = np.array(tracked_errors)
# tracked_errors_std = np.array(tracked_errors_std)
# tracked_percentile1 = np.array(tracked_percentile1)
# tracked_percentile2 = np.array(tracked_percentile2)

# # comment out if you dont want to rerun and use only pickles
# with open("pickle_files/new_pickles_"+dataset_name+"_"+similarity_measure+".pkl", "wb") as pickle_file:
#     pickle.dump([sample_eigenvalues_scaled, sample_eigenvalues_scaled_std, \
#             tracked_errors, tracked_errors_std, tracked_percentile1, tracked_percentile2], pickle_file)

# uncomment to load from pickle file only
with open("pickle_files/new_pickles_"+dataset_name+"_"+similarity_measure+".pkl", "rb") as pickle_file:
    A = pickle.load (pickle_file)
[sample_eigenvalues_scaled, sample_eigenvalues_scaled_std, \
                  tracked_errors, tracked_errors_std, \
                  tracked_percentile1, tracked_percentile2] = A

# uncomment to load eigvalues from saved file
f = open("figures/"+dataset_name+"/eigvals.txt", "r")
all_lines = f.readlines()
f.close()
all_lines = [x.strip("\r\b") for x in all_lines]
for i in range(len(all_lines)):
    val_, index_ = all_lines[i].split()
    val_ = float(val_)
    index_ = int(index_)
    true_spectrum[index_] = val_
chosen_eig = true_spectrum[search_rank]

################################################################################################

for i in range(len(search_rank)):
    display(dataset_name, similarity_measure, true_spectrum, dataset_size, search_rank[i], \
        sample_eigenvalues_scaled[:,i], sample_eigenvalues_scaled_std[:,i], max_samples, min_samples)
    display_precomputed_error(dataset_name, similarity_measure, tracked_errors[:,i], \
        dataset_size, search_rank[i], max_samples, \
        error_std=tracked_errors_std[:,i],\
        percentile1=tracked_percentile1[:,i], \
        percentile2=tracked_percentile2[:,i], min_samples=min_samples)
    # display_precomputed_error(dataset_name, similarity_measure, tracked_errors[:,i], \
    #     dataset_size, search_rank[i], max_samples, \
    #     error_std=tracked_errors_std[:,i],\
    #     percentile1=tracked_percentile1[:,i], \
    #     percentile2=tracked_percentile2[:,i], log = False)
