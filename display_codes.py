"""
visualization codes
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def display_image(image):
    plt.gcf().clear()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(image, cmap="gray")
    plt.xlabel("x co-ordinates", fontsize=16)
    plt.ylabel("y co-ordinates", fontsize=16)
    
    plt.title("Original image", fontsize=16)
    filename = "./figures/kong/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+"kong-original.pdf"
    plt.savefig(filename)
    return None
    pass

def display_kong_dataset(xy):
    plt.gcf().clear()
    plt.rcParams.update({'font.size': 12})
    plt.scatter(xy[:,1], xy[:,0], 0.3, color="#029386")
    plt.xlabel("x co-ordinates", fontsize=16)
    plt.ylabel("y co-ordinates", fontsize=16)
    
    plt.title("Sampled points", fontsize=18)
    filename = "./figures/kong/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+"kong-subsamples.pdf"
    plt.savefig(filename)
    return None

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

    true_min_eig_vec = true_min_eig*np.ones_like(x_axis)
    print(true_min_eig, search_rank)

    estimate_min_eig_vec = np.array(sample_eigenvalues_scaled)
    estimate_std = np.array(sample_eigenvalues_scaled_std)

    plt.gcf().clear()
    plt.rcParams.update({'font.size': 11})
    plt.plot(x_axis, true_min_eig_vec, label="True eigenvalue", alpha=1.0, color="#15B01A")
    plt.plot(x_axis, estimate_min_eig_vec, label="Approximate eigenvalue", alpha=1.0, color="#FC5A50")
    plt.fill_between(x_axis, estimate_min_eig_vec-estimate_std, estimate_min_eig_vec+estimate_std, alpha=0.2, color="#FC5A50")
    plt.xlabel("Proportion of dataset chosen as landmark samples")
    plt.ylabel("Eigenvalue estimates")
    plt.legend(loc="upper right")
    
    # title of the file
    if similarity_measure == "ht":
        plt.title("Hyperbolic: "+convert_rank_to_order(search_rank)+" eigenvalue")
    if similarity_measure == "tps":
        plt.title("TPS: "+convert_rank_to_order(search_rank)+" eigenvalue")
    if similarity_measure == "default":
        if dataset_name == "arxiv":
            plt.title("ArXiv: "+convert_rank_to_order(search_rank)+" eigenvalue")
        else:
            plt.title(dataset_name.capitalize()+": "+convert_rank_to_order(search_rank)+" eigenvalue")
    filename = "./figures/"+dataset_name+"/eigenvalues/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+similarity_measure+"_"+str(search_rank)+".pdf"
    plt.savefig(filename)
    return None


def display_precomputed_error(dataset_name, similarity_measure, error, dataset_size, \
                              search_rank, max_samples, error_std=[], \
                              percentile1=[], percentile2=[], log=True, min_samples=50, true_eigval=0):
    np.set_printoptions(precision=2)
                              # percentile1=[], percentile2=[], log=True, min_samples=50):
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    size_of_fonts = 16

    np.set_printoptions(precision=0)
    x_axis = np.array(list(range(min_samples, max_samples, 10))) / dataset_size
    x_axis = np.log(x_axis)

    plt.gcf().clear()
    fig, ax = plt.subplots()

    if dataset_name == "erdos" and search_rank != -1:
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    plt.rcParams.update({'font.size': 18})

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
    plt.legend(loc="upper right")
    
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
                if dataset_name == "synthetic_tester":
                    plt.title(similarity_measure+": "+convert_rank_to_order(search_rank)+" eigenvalue = "+str(true_eigval))
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

def frobenius_error_disp(mean_errors, std_errors, dataset_name, min_samples, max_samples, step_samples, dataset_size):
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    size_of_fonts = 16

    np.set_printoptions(precision=0)
    x_axis = np.array(list(range(min_samples, max_samples, step_samples))) / dataset_size
    x_axis = np.log(x_axis)

    plt.gcf().clear()
    fig, ax = plt.subplots()

    plt.rcParams.update({'font.size': 18})
    plt.plot(x_axis, np.log(mean_errors), alpha=1.0, color="#069AF3")
    plt.fill_between(x_axis, np.log(mean_errors-std_errors), np.log(mean_errors+std_errors), alpha=0.2, color="#069AF3")

    plt.xlabel("log sampling rate", fontsize = size_of_fonts)
    plt.ylabel("log average frobenius error", fontsize = size_of_fonts)

    plt.title("Random matrix")
    
    # save file
    filename = "./figures/"+dataset_name+"/frobenius_error/"
    if not os.path.isdir(filename):
        os.makedirs(filename)
    filename = filename+"error.pdf"
    plt.savefig(filename)

    return None
