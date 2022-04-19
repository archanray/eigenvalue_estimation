import numpy as np
from src.display_codes import display_combined_error
import matplotlib.pyplot as plt

def plot_all_errors(tracked_errors, tracked_percentile1, tracked_percentile2, \
            sampling_modes, dataset_name, dataset_size, \
            search_rank, max_samples, min_samples, \
            true_spectrum, name_adder):

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
        display_combined_error(sampling_modes, dataset_name, tracked_errors_rank, \
            dataset_size, search_rank[i], max_samples, \
            percentile1=percentile1_rank, \
            percentile2=percentile2_rank, min_samples=min_samples, \
            true_eigval=true_spectrum[search_rank[i]], name_adder=name_adder)


def plot_all_nnz(nnz_sm, sampling_modes, dataset_name):
    plt.gcf().clear()
    for m in sampling_modes:
        if "sparsity sampler" in m:
            plt.plot(nnz_sm[m], label=m)
        
    plt.legend(loc="upper right")
    plt.savefig("./figures/nnzs/"+dataset_name+"_sparsity.pdf")