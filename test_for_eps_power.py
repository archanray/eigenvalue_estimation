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
from display_codes import convert_rank_to_order

def display_precomputed_error(dataset_name, plot_data):
    """
    display functions I need for visualization
    """
    color_palette = \
            ["#FF796C", "#9ACD32", "#06C2AC", "#069AF3", "#DAA520", "#4B0082", "#C875C4"]
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
    x_limit = np.argmin(x_axis <= -1.3)

    for r in range(len(search_ranks)):
        plt.gcf().clear()
        fig, ax = plt.subplots()

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        plt.rcParams.update({'font.size': 14})

        for i in range(len(eps_pows)):
            # get the data from the savelog
            tracked_error_means = np.array(eps_means_per_round[i])
            tracked_percentile1 = np.array(eps_percent1_per_round[i])
            tracked_percentile2 = np.array(eps_percent2_per_round[i])

            error_means = tracked_error_means[:, r]
            error_p1 = tracked_percentile1[:,r]
            error_p2 = tracked_percentile2[:,r]

            c = color_palette[i]
            # finally plot this!
            plt.plot(x_axis[:x_limit+1], np.log(error_means[:x_limit+1]), \
                label="eps^"+str(eps_pows[i]), alpha=1.0, color=c)
            plt.fill_between(x_axis[:x_limit+1], np.log(error_p1[:x_limit+1]), np.log(error_p2[:x_limit+1]), alpha=0.2, color=c)
        
        plt.ylabel("Log of average scaled absolute error", fontsize=size_of_fonts)
        plt.xlabel("Log sampling rate", fontsize=size_of_fonts)
    
        # title of the file
        plt.title("Comparison of errors for "+\
            convert_rank_to_order(search_ranks[r])+" eigenvalue", fontsize=size_of_fonts)
        plt.legend(loc="upper right", fontsize=0.7*size_of_fonts)
    
        # save the file
        filename = "./figures/"+dataset_name+"_eps_varied"+"/errors/"
        if not os.path.isdir(filename):
            os.makedirs(filename)
        filename = filename+"_"+str(search_ranks[r])+".pdf"
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

################################### COMPUTE ERRORS AND EIGS #####################################
def run_all(x):
    # logging data-structures
    # set power of eps and loop around it
    eps_means_per_round = []
    eps_percent1_per_round = []
    eps_percent2_per_round = []

    # comment out if you dont want to rerun and use only pickles
    for pows in x.eps_pows:
        print(pows)
        error_val_means = []
        percentile1 = []
        percentile2 = []
        for s in tqdm(range(x.min_samples, x.max_samples, x.steps)):
            # compute the eps for the given s: s=1/eps^pows ==> eps = 1/s^{1/pows}
            local_eps = 1 / (s ** (1/pows))
            
            # run experiment trials        
            error_vals = []
            # create the matrix 
            matrix, n, _, _ = get_data(x.dataset_name, eps=local_eps, plot_mat=False, raise_eps=True)
            # get the true spectrum and the subset we are guning for
            true_spectrum = np.real(np.linalg.eigvals(matrix))
            true_spectrum.sort()
            chosen_eig = true_spectrum[x.search_rank]

            for j in range(x.trials):
                ## create the matrix 
                # matrix, n, _, _ = get_data(dataset_name, eps=local_eps, plot_mat=False)

                # # get the true spectrum and the subset we are guning for
                # true_spectrum = np.real(np.linalg.eigvals(matrix))
                # true_spectrum.sort()
                # chosen_eig = true_spectrum[search_rank]

                # compute the approximate eigenvalues
                min_eig_single_round = \
                    sample_eig_default(matrix, s, True, rankcheck=x.search_rank)
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

    with open("pickle_files/multi_eps_new_pickles_"+x.dataset_name+".pkl", "wb") as pickle_file:
        pickle.dump([eps_means_per_round, eps_percent1_per_round, \
                        eps_percent2_per_round, x.min_samples, \
                        x.max_samples, x.steps, x.eps_pows, n, x.search_rank], pickle_file)

    return None

################################ LOAD PRECOMPUTED DATA AND PLOT ##############################
def plot_only(x):
    with open("pickle_files/multi_eps_new_pickles_"+x.dataset_name+".pkl", "rb") as pickle_file:
        plot_data = pickle.load(pickle_file)
    display_precomputed_error(x.dataset_name, plot_data)
    return None

##############################################################################################
# parameters to retrieve dataset
class Variables:
    def __init__(self):
        self.trials = 50
        self.search_rank = [0,1,2,3,-4,-3,-2,-1]
        self.dataset_name = "multi_block_outer"
        self.min_samples = 50
        self.max_samples = 500
        self.steps = 10
        self.eps_pows = [1.5, 2, 2.5, 3, 3.5, 4]
        self.run_mode = "plot"

runner = Variables()

if runner.run_mode == "plot":
    plot_only(runner)

if runner.run_mode == "full":
    run_all(runner)
#################################################################################################
