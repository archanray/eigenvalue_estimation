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

# A very fast distance function

def get_distance(X, Y):
    """
    fast euclidean distance computation:
    X shape is n x features
    """
    num_test = X.shape[0]
    num_train = Y.shape[0]
    dists = np.zeros((num_test, num_train))
    sum1 = np.sum(np.power(X,2), axis=1)
    sum2 = np.sum(np.power(Y,2), axis=1)
    sum3 = 2*np.dot(X, Y.T)
    dists = sum1.reshape(-1,1) + sum2
    dists = np.sqrt(dists - sum3)
    dists = dists / np.max(dists)
    return dists

# Get dataset function
def get_data(name, eps=0.1, plot_mat=True, raise_eps=False, max_size=1000):
  if name == "MNIST":
        """
        get MNIST dataset from: http://yann.lecun.com/exdb/mnist/
        dataset: MNIST 10k
        do pip install idx2numpy prior to this
        """
        import idx2numpy
        file = 't10k-images.idx3-ubyte'
        arr = idx2numpy.convert_from_file(file)
        # reshape images to vectors: the following line leads to array of size 10k x 784
        arr = arr.reshape(arr.shape[0], arr.shape[1]*arr.shape[2])

        ## subsample to fit RAM
        sample_indices = np.sort(random.sample(list(range(len(arr))), max_size))
        arr = arr[sample_indices]
        A = get_distance(arr, arr)
        return A, len(A), 50, 1000

  if name == "block":
        """
        uses a matrix with n/2 x n/2 block of all 1s and rest zeros
        """
        dataset_size = 5000
        A = np.zeros((dataset_size, dataset_size))
        B = np.ones((int(dataset_size/2), int(dataset_size/2)))
        A[0:len(B), 0:len(B)] = B

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return A, dataset_size, min_sample_size, max_sample_size

  if name == "erdos":
        from networkx.generators.random_graphs import erdos_renyi_graph
        n = max_size
        g = erdos_renyi_graph(n, p=0.7)
        A = nx.adjacency_matrix(g)
        A = A.todense()
        return A, n, int(n/100), int(n/5)

  if name == "facebook":
        data_file = "./data/facebook_combined.txt" 
        # import networkx as nx
        g = nx.read_edgelist(data_file,create_using=nx.DiGraph(), nodetype = int)
        A = nx.adjacency_matrix(g)
        A = A.todense()
        A = A+A.T
        n = len(A)
        return A, n, int(n/100), int(n/5)


# The eigenvalue estimator
def sample_eig_default(data_matrix, s, scale=False, rankcheck=0, mode="random_sample", norm=[]):
    """
    input: original matrix
    output: sample eigenvalue
    requires data matrix to be fully instantiated
    """
    n = len(data_matrix)
    list_of_available_indices = range(n)
    if mode == "random_sample":
        norm = np.ones(len(data_matrix)) / n
        # sample_indices = np.sort(random.sample(list_of_available_indices, s))
        sample_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=True, p=norm))
        chosen_p = norm[sample_indices]
        subsample_matrix = data_matrix[sample_indices][:, sample_indices]
        # reweight using pj/s
        sqrt_chosen_p = np.sqrt(chosen_p)/s
        D = np.diag(sqrt_chosen_p)
        subsample_matrix = D @ subsample_matrix @ D
    
    if mode == "CUR":
        """
        uses CUR's U to compute estimates in sublinear time
        based on 5.1.1 of https://www.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/5-cur.pdf
        """
        n = len(data_matrix)
        list_of_available_indices = range(n)
        sample_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=False))
        # select columns
        C = data_matrix[sample_indices].T / np.sqrt(s*1/n) # since p_j is 1/n
        # select rows and columns; we don't need to compute R since we are interested in U only
        psi = data_matrix[sample_indices][:, sample_indices] / (s*1/n)
        # compute U
        try:
        	subsample_matrix = np.linalg.inv(C.T @ C) @ psi.T
        except:
        	subsample_matrix = np.linalg.inv(C.T @ C + 1e-10*np.eye(len(C.T))) @ psi.T
      
    if mode == "separate_CUR":
        n = len(data_matrix)
        list_of_available_indices = range(n)
        sample_row_indices = np.sort(random.sample(list_of_available_indices, s))
        sample_col_indices = np.sort(random.sample(list_of_available_indices, s))
        # select columns
        C = data_matrix[sample_col_indices].T / np.sqrt(s*1/n) # since p_j is 1/n
        # select rows and columns; we don't need to compute R since we are interested in U only
        psi = data_matrix[sample_row_indices][:, sample_col_indices] / (s*1/n)
        # compute U
        try:
        	subsample_matrix = np.linalg.inv(C.T @ C) @ psi.T
        except:
        	subsample_matrix = np.linalg.inv(C.T @ C + 1e-10*np.eye(len(C.T))) @ psi.T

    if mode == "norm_CUR":
        n = len(data_matrix)
        list_of_available_indices = range(n)
        sample_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=False, p=norm))
        # select columns
        C = data_matrix[sample_indices].T / np.sqrt(s*1/n) # since p_j is 1/n
        # select rows and columns; we don't need to compute R since we are interested in U only
        psi = data_matrix[sample_indices][:, sample_indices] / (s*1/n)
        # compute U
        try:
        	subsample_matrix = np.linalg.inv(C.T @ C) @ psi.T
        except:
        	subsample_matrix = np.linalg.inv(C.T @ C + 1e-10*np.eye(len(C.T))) @ psi.T

    if mode == "separate_norm_CUR":
        n = len(data_matrix)
        list_of_available_indices = range(n)
        sample_row_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=False, p=norm))
        sample_col_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=False, p=norm))
        # select columns
        C = data_matrix[sample_col_indices].T / np.sqrt(s*1/n) # since p_j is 1/n
        # select rows and columns; we don't need to compute R since we are interested in U only
        psi = data_matrix[sample_row_indices][:, sample_col_indices] / (s*1/n)
        # compute U
        try:
        	subsample_matrix = np.linalg.inv(C.T @ C) @ psi.T
        except:
        	subsample_matrix = np.linalg.inv(C.T @ C + 1e-10*np.eye(len(C.T))) @ psi.T

    if mode == "norm":
        sample_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=True, p=norm))
        chosen_p = norm[sample_indices]
        subsample_matrix = data_matrix[sample_indices][:, sample_indices]
        # compute Ds
        sqrt_chosen_p = np.sqrt(chosen_p)/s
        D = np.diag(sqrt_chosen_p)
        subsample_matrix = D @ subsample_matrix @ D
    
    if mode == "separate":
        sample_row_indices = np.sort(random.sample(list_of_available_indices, s))
        sample_col_indices = np.sort(random.sample(list_of_available_indices, s))
        subsample_matrix = data_matrix[sample_row_indices][:, sample_col_indices, ]
    
    if mode == "separate_norm":
        sample_row_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=False, p=norm))
        sample_col_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=False, p=norm))
        subsample_matrix = data_matrix[sample_row_indices][:, sample_col_indices, ]

    # useful for only hermitian matrices
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()

    min_eig = np.array(all_eig_val_estimates)[rankcheck]
    if scale == False or "CUR" in mode:
        return min_eig
    else:
        return n*min_eig/float(s)


# Code for visualization
def display_precomputed_error(dataset_name, error, dataset_size, \
                              search_rank, max_samples, \
                              percentile1=[], percentile2=[], min_samples=50, true_eigval=0):
    np.set_printoptions(precision=2)
                              # percentile1=[], percentile2=[], log=True, min_samples=50):
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    size_of_fonts = 14

    np.set_printoptions(precision=0)
    x_axis = np.array(list(range(min_samples, max_samples, 10))) / dataset_size
    x_axis = np.log(x_axis)

    plt.gcf().clear()
    fig, ax = plt.subplots()

    if dataset_name == "erdos" and search_rank != -1:
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    plt.rcParams.update({'font.size': 16})

    plt.plot(x_axis, np.log(error), label="log of average scaled absolute error", alpha=1.0, color="#069AF3")
    plt.fill_between(x_axis, np.log(percentile1), np.log(percentile2), alpha=0.2, color="#069AF3")
    plt.ylabel("Log of average scaled absolute error", fontsize=size_of_fonts)
    
    plt.xlabel("Log sampling rate", fontsize=size_of_fonts)
    if dataset_name == "block" and search_rank == -1:
        plt.ylim(-6.0, -2.5)
    # plt.legend(loc="upper right")
    
    # title of the file
    plt.title(dataset_name.capitalize()+": "+convert_rank_to_order(search_rank)+" eigenvalue")
    
    # save the file
    filename = "./figures/"+dataset_name+"/errors/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+"_"+str(search_rank)+".pdf"
    # uncomment to save file
    # plt.savefig(filename)
    # uncomment to visualize file
    plt.show()
    # from google.colab import files
    # files.download(filename)
    
    return None

# Code for some naming fixing in the visualization code
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

# display precomputed error
def display_precomputed_error(dataset_name, error, dataset_size, \
                              search_rank, max_samples, \
                              percentile1=[], percentile2=[], min_samples=50, true_eigval=0,\
                              name_adder="default"):
    np.set_printoptions(precision=2)
                              # percentile1=[], percentile2=[], log=True, min_samples=50):
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    size_of_fonts = 14

    np.set_printoptions(precision=0)
    x_axis = np.array(list(range(min_samples, max_samples, 10))) / dataset_size
    x_axis = np.log(x_axis)

    plt.gcf().clear()
    fig, ax = plt.subplots()

    if dataset_name == "erdos" and search_rank != -1:
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    plt.rcParams.update({'font.size': 13})
    number_of_plots = len(sampling_modes)
    colormap = plt.cm.nipy_spectral
    # colors = [colormap(i) for i in np.linspace(0, 1,number_of_plots)]
    # print(colors)
    # ax.set_prop_cycle('color', colors)
    
    for m in sampling_modes:
        plt.plot(x_axis, np.log(error[m]), label="log of average scaled absolute error", alpha=1.0)
        plt.fill_between(x_axis, np.log(percentile1[m]), np.log(percentile2[m]), alpha=0.2)
        plt.ylabel("Log of average scaled absolute error", fontsize=size_of_fonts)
    plt.legend(sampling_modes)
    plt.xlabel("Log sampling rate", fontsize=size_of_fonts)

    if dataset_name == "block" and search_rank == -1:
        plt.ylim(-6.0, -2.5)
    # plt.legend(loc="upper right")
    
    # title of the file
    plt.title(dataset_name.capitalize()+": "+convert_rank_to_order(search_rank)+" eigenvalue")
    
    # save the file
    if name_adder == "default":
    	filename = "./figures/"+dataset_name+"/errors/"
    else:
    	filename = "./figures/"+dataset_name+"_"+name_adder+"/errors/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+"_"+str(search_rank)+".pdf"
    # uncomment to visualize file
    #plt.show()
    # uncomment to download file
    plt.savefig(filename)
    
    return None

# Parameters
trials = 50
# similarity_measure = "default"#"tps" #"tps", "ht" for kong, "default" for binary and random_sparse
search_rank = [0,1,2,3,-4,-3,-2,-1]
max_size = 5000
dataset_name = "erdos"
# dataset_name = "erdos", "MNIST", "block", "facebook"
name_adder = "norm_v_random"
# sampling_modes = ["random_sample", "separate", "norm", "separate_norm"]
# sampling_modes = ["random_sample", "CUR"]
# sampling_modes = ["CUR", "separate_CUR", "norm_CUR", "separate_norm_CUR"]
sampling_modes = ["random_sample", "norm"]

# Get the dataset
true_mat, dataset_size, min_samples, max_samples = get_data(dataset_name, max_size=max_size)
# print(true_mat.shape)

#select duplicate row and col
unq, count = np.unique(true_mat, axis=0, return_counts=True)
repeated_groups = unq[count > 1]
unq = unq[count == 1]
# print(unq.shape)
# print(len(repeated_groups))
#"""
# uncommment when running the full code
true_spectrum = np.real(np.linalg.eigvals(true_mat))

# uncomment when running from saved values
# true_spectrum = np.zeros(len(true_mat))
#"""
print("loaded dataset")
print("||A||_infty:", np.max(true_mat))

# set up output loggers
tracked_errors = {}
tracked_errors_std = {}
tracked_percentile1 = {}
tracked_percentile2 = {}
true_spectrum.sort()
chosen_eig = true_spectrum[search_rank]
norm = np.linalg.norm(true_mat, axis=1)**2 / np.linalg.norm(true_mat)**2
for m in sampling_modes:
    tracked_errors[m] = []
    tracked_errors_std[m] = []
    tracked_percentile1[m] = []
    tracked_percentile2[m] = []

# plot row norms
print(norm.shape)
plt.hist(norm, density=False, bins=30)
plt.xlabel("Data")
plt.ylabel("Probability")
plt.savefig("./figures/"+dataset_name+"_row_norm.pdf")

#"""
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
            if m == "norm" or m == "separate_norm" or m == "norm_CUR" or m == "separate_norm_CUR":
                min_eig_single_round = sample_eig_default(true_mat, i, True, 
                                                          rankcheck=search_rank, mode=m, norm=norm)
            else:
                min_eig_single_round = sample_eig_default(true_mat, i, True,
                                                          rankcheck=search_rank,
                                                          mode=m)
            # get error this round
            error_single_round = np.abs(min_eig_single_round - chosen_eig) / float(dataset_size)
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
    display_precomputed_error(dataset_name, tracked_errors_rank, \
        dataset_size, search_rank[i], max_samples, \
        percentile1=percentile1_rank, \
        percentile2=percentile2_rank, min_samples=min_samples, \
        true_eigval=true_spectrum[search_rank[i]], name_adder=name_adder)
#"""
