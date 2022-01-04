"""
code to test different powers of epsilon
"""

import numpy as np
import random
from tqdm import tqdm
from get_dataset import get_data
from similarities import hyperbolic_tangent, thin_plane_spline
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator, MultipleLocator

def convert_rank_to_order(search_rank):
    """
    convert numbers to names (preordained and not ordinal replacements)
    """
    if search_rank == 0:
        rank_name = "smallest"
    if search_rank == 1:
        rank_name = "second smallest"
    if search_rank == 2:
        rank_name = "third smallest"
    if search_rank == 3:
        rank_name = "fourth smallest"
    if search_rank == -1:
        rank_name = "largest"
    if search_rank == -2:
        rank_name = "second largest"
    if search_rank == -3:
        rank_name = "third largest"
    if search_rank == -4:
        rank_name = "fourth largest"

    return rank_name

def display_precomputed_error(dataset_name, plot_data):
    """
    display functions I need for visualization
    """
    eps_means_per_round = plot_data[0]
    eps_percent1_per_round = plot_data[1]
    eps_percent2_per_round = plot_data[2]
    min_samples = plot_data[3]
    max_samples = plot_data[4]
    steps = plot_data[5]
    eps_pows = plot_data[6]
    dataset_size = plot_data[7]
    search_ranks = plot_data[8]

    np.set_printoptions(precision=2)
    size_of_fonts = 14

    np.set_printoptions(precision=0)
    x_axis = np.array(list(range(min_samples, max_samples, 10))) / dataset_size
    x_axis = np.log(x_axis)

    for search_rank in search_ranks:
        plt.gcf().clear()
        fig, ax = plt.subplots()

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        plt.rcParams.update({'font.size': 16})

        for pows in eps_pows:
            pass
        pass

    if log == True:
        plt.plot(x_axis, np.log(error), label="log of average scaled absolute error", alpha=1.0, color="#069AF3")
    else:
        plt.plot(x_axis, error, label="average absolute error", alpha=1.0, color="#069AF3")
    if percentile1 == []:
        plt.fill_between(x_axis, np.log(error-error_std), np.log(error+error_std), alpha=0.2, color="#069AF3")
        plt.ylabel("Log of average scaled absolute error")
        pass
    else:
        if log == True:
            plt.fill_between(x_axis, np.log(percentile1), np.log(percentile2), alpha=0.2, color="#069AF3")
            plt.ylabel("Log of average scaled absolute error", fontsize=size_of_fonts)
        else:
            plt.fill_between(x_axis, percentile1, percentile2, alpha=0.2, color="#069AF3")
            plt.ylabel("average scaled absolute error of eigenvalue estimates")

    plt.xlabel("Log sampling rate", fontsize=size_of_fonts)
    
    if dataset_name == "block" and search_rank == -1:
        plt.ylim(-6.0, -2.5)
    # plt.legend(loc="upper right")
    
    # title of the file
    if similarity_measure == "ht":
        plt.title("Hyperbolic: "+convert_rank_to_order(search_rank)+" eigenvalue")
    if similarity_measure == "tps":
        plt.title("TPS: "+convert_rank_to_order(search_rank)+" eigenvalue")
    if similarity_measure == "default":
        if dataset_name == "arxiv":
            plt.title("ArXiv: "+convert_rank_to_order(search_rank)+" eigenvalue")
        else:
            if dataset_name == "erdos":
                plt.title("ER: "+convert_rank_to_order(search_rank)+" eigenvalue")
            else:
                if dataset_name == "synthetic_tester" or dataset_name == "multi_block_synthetic" or dataset_name == "multi_block_outer":
                    plt.title(convert_rank_to_order(search_rank)+" eigenvalue = "+str(round(true_eigval,2)))
                else:
                    plt.title(dataset_name.capitalize()+": "+convert_rank_to_order(search_rank)+" eigenvalue")
    
    # save the file
    if log == True:
        filename = "./figures/"+dataset_name+"/errors/"
    else:
        filename = "./figures/"+dataset_name+"/non_log_errors/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+similarity_measure+"_"+str(search_rank)+".pdf"
    plt.savefig(filename)
    
    return None

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
steps = 10
#################################################################################################

################################### COMPUTE ERRORS AND EIGS #####################################
# logging data-structures
# set power of eps and loop around it
eps_pows = [1.5, 2, 2.5, 3]
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
        # create the matrix 
        matrix, n, _, _ = get_data(dataset_name, eps=local_eps, plot_mat=False)
        # get the true spectrum and the subset we are guning for
        true_spectrum = np.real(np.linalg.eigvals(matrix))
        true_spectrum.sort()
        chosen_eig = true_spectrum[search_rank]

        for j in range(trials):
            ## create the matrix 
            # matrix, n, _, _ = get_data(dataset_name, eps=local_eps, plot_mat=False)

            # # get the true spectrum and the subset we are guning for
            # true_spectrum = np.real(np.linalg.eigvals(matrix))
            # true_spectrum.sort()
            # chosen_eig = true_spectrum[search_rank]

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
    pickle.dump([eps_means_per_round, eps_percent1_per_round, \
                    eps_percent2_per_round, min_samples, max_samples, steps, eps_pows, n, search_rank], pickle_file)