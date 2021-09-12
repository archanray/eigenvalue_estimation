import matplotlib.pyplot as plt
import numpy as np
import os


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

def display(dataset_name, similarity_measure, true_eigvals, dataset_size, search_rank, \
            sample_eigenvalues_scaled, sample_eigenvalues_scaled_std, max_samples, min_samples):
    true_min_eig = true_eigvals[search_rank]

    x_axis = np.array(list(range(min_samples, max_samples, 10))) / dataset_size
    # clip all samples under 50
    # x_axis = x_axis[4:]

    true_min_eig_vec = true_min_eig*np.ones_like(x_axis)
    print(true_min_eig, search_rank)

    estimate_min_eig_vec = np.array(sample_eigenvalues_scaled)
    estimate_std = np.array(sample_eigenvalues_scaled_std)
    # clip all samples under 50
    # estimate_min_eig_vec = estimate_min_eig_vec[4:]
    # estimate_std = estimate_std[4:]

    plt.gcf().clear()
    plt.plot(x_axis, true_min_eig_vec, label="True", alpha=1.0, color="#15B01A")
    plt.plot(x_axis, estimate_min_eig_vec, label="Scaled estimate", alpha=1.0, color="#FC5A50")
    plt.fill_between(x_axis, estimate_min_eig_vec-estimate_std, estimate_min_eig_vec+estimate_std, alpha=0.2, color="#FC5A50")
    plt.xlabel("Proportion of dataset chosen as landmark samples")
    plt.ylabel("Eigenvalue estimates")
    plt.legend(loc="upper right")
    
    plt.title(similarity_measure+": "+convert_rank_to_order(search_rank)+" eigenvalue")
    filename = "./figures/"+dataset_name+"/eigenvalues/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+similarity_measure+"_"+str(search_rank)+".pdf"
    plt.savefig(filename)
    return None


def display_precomputed_error(dataset_name, similarity_measure, error, dataset_size, \
                              search_rank, max_samples, error_std=[], \
                              percentile1=[], percentile2=[], log=True, min_samples=50):
    np.set_printoptions(precision=2)
    x_axis = np.array(list(range(min_samples, max_samples, 10))) / dataset_size
    # clip all samples under 50
    # x_axis = x_axis[4:]
    x_axis = np.log(x_axis)
    # clip all samples under 50
    # error = error[4:]
    if error_std != []:
        # error_std = error_std[4:]
        pass

    plt.gcf().clear()
    # plt.plot(x_axis, error, label="log of relative absolute error", alpha=1.0, color="#069AF3")
    if log == True:
        plt.plot(x_axis, np.log(error), label="log of average absolute error", alpha=1.0, color="#069AF3")
    else:
        plt.plot(x_axis, error, label="average absolute error", alpha=1.0, color="#069AF3")
    if percentile1 == []:
        plt.fill_between(x_axis, np.log(error-error_std), np.log(error+error_std), alpha=0.2, color="#069AF3")
        plt.ylabel("Log of scaled average absolute error of eigenvalue estimates")
        pass
    else:
        if log == True:
            plt.fill_between(x_axis, np.log(percentile1), np.log(percentile2), alpha=0.2, color="#069AF3")
            plt.ylabel("Log of scaled average absolute error of eigenvalue estimates")
        else:
            plt.fill_between(x_axis, percentile1, percentile2, alpha=0.2, color="#069AF3")
            plt.ylabel("scaled average absolute error of eigenvalue estimates")

    plt.xlabel("Log of proportion of dataset chosen as landmark samples")
    # plt.ylabel("Log of relative absolute error of eigenvalue estimates")
    if dataset_name == "block" and search_rank == -1:
        plt.ylim(-6.0, -2.5)
    plt.legend(loc="upper right")
    plt.title(similarity_measure+": "+convert_rank_to_order(search_rank)+" eigenvalue")
    if log == True:
        filename = "./figures/"+dataset_name+"/errors/"
    else:
        filename = "./figures/"+dataset_name+"/non_log_errors/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+similarity_measure+"_"+str(search_rank)+".pdf"
    plt.savefig(filename)
    return None


def display_error_percentile(dataset_name, similarity_measure, error, dataset_size,\
                            search_rank, max_samples, percentile1, percentile2):
    np.set_printoptions(precision=2)
    x_axis = np.array(list(range(50, max_samples, 10))) / dataset_size
    x_axis = np.log(x_axis)
    plt.gcf().clear()
    plt.plot(x_axis, np.log(error), label="log of average absolute error", alpha=1.0, color="#069AF3")
    plt.fill_between(x_axis, np.log(percentile1), np.log(percentile2), alpha=0.2, color="#069AF3")
    plt.ylabel("Log of scaled average absolute error of eigenvalue estimates")
    plt.xlabel("Log of proportion of dataset chosen as landmark samples")

    if dataset_name == "block" and search_rank == -1:
        plt.ylim(-6.0, -2.5)
    plt.legend(loc="upper right")
    plt.title(similarity_measure+": "+convert_rank_to_order(search_rank)+" eigenvalue")
    filename = "./figures/"+dataset_name+"/errors/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+similarity_measure+"_"+str(search_rank)+".pdf"
    plt.savefig(filename)
    return None