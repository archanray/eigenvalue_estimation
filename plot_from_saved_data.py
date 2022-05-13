import pickle
from src.viz import plot_all_errors, plot_all_nnz

dataset_name = "block"
name_adder = "uniform"

with open("pickle_files/"+dataset_name+"_"+name_adder+".pkl", "rb") as f:
    [tracked_errors, tracked_percentile1, tracked_percentile2, sampling_modes, dataset_name, dataset_size, \
            search_rank, max_samples, min_samples, \
            true_spectrum, name_adder] = pickle.load(f)

# visualize the errors
plot_all_errors(tracked_errors, tracked_percentile1, tracked_percentile2, \
            sampling_modes, dataset_name, dataset_size, \
            search_rank, max_samples, min_samples, \
            true_spectrum, name_adder)