import matplotlib.pyplot as plt
import numpy as np
import os

def display(dataset_name, similarity_measure, true_eigvals, dataset_size, search_rank, \
            sample_eigenvalues_scaled):
    true_eigvals.sort()
    true_min_eig = true_eigvals[search_rank]

    x_axis = np.array(list(range(10, 1000, 10))) / dataset_size
    true_min_eig_vec = true_min_eig*np.ones_like(x_axis)

    estimate_min_eig_vec = np.array(sample_eigenvalues_scaled)

    plt.plot(x_axis, true_min_eig_vec, label="True", alpha=0.5)
    plt.plot(x_axis, estimate_min_eig_vec, label="Scaled estimate", alpha=0.5)
    plt.xlabel("Proportion of dataset chosen as landmark samples")
    plt.ylabel("Eigenvalue estimates")
    plt.legend(loc="upper right")
    plt.title(similarity_measure+": "+str(search_rank)+"th eigenvalue")
    filename = "./figures/dataset_name/eigenvalues/"
    if not os.isdir(filename):
        os.path.mkdirs(filename)
    filename = filename+similarity_measure+"_"+str(search_rank)+".pdf"
    plt.savefig(filename)
    return None


def display_precomputed_error(dataset_name, similarity_measure, error, dataset_size, \
                              search_rank, sample_eigenvalues_scaled):
    x_axis = np.array(list(range(10, 1000, 10))) / dataset_size
    plt.plot(x_axis, error, label="log of squared error", alpha=0.5)
    plt.xlabel("Proportion of dataset chosen as landmark samples")
    plt.ylabel("Error of eigenvalue estimates")
    plt.legend(loc="upper right")
    plt.title(similarity_measure+": "+str(search_rank)+"th eigenvalue")
    filename = "./figures/dataset_name/errors/"
    if not os.isdir(filename):
        os.path.mkdirs(filename)
    filename = filename+similarity_measure+"_"+str(search_rank)+".pdf"
    plt.savefig(filename)
    return None
