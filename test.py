import numpy as np
import random
from tqdm import tqdm
from display_codes import display, display_precomputed_error
from get_dataset import get_data
from similarities import sigmoid, tps
import pickle

# labels and other hyperparameters
dataset_name = "kong"
# similarity_measure = "tps"
search_rank = [-3]

# load dataset
dataset, n = get_data(dataset_name)
similarity_measure = tps
true_matrix = tps(dataset, dataset)
true_eig = np.linalg.eigvalsh(true_matrix)[-3]

per_size_eps_mean = []
per_size_eps_max = []
per_size_eps_min = []
per_size_eps_std = []
per_size_eps_10p = []
per_size_eps_90p = []

print("starting approximation")

# perform approximation
for sample_size in tqdm(range(50, 1000, 50)):
    # approx_eigval_list = []
    per_round_eps = []
    for trials in range(100):
        print(trials)
        all_ids = np.random.permutation(n)
        samples = all_ids[0:n]
        sample_matrix = true_matrix[samples][:,samples]
        approx_eigval = n * np.linalg.eigvalsh(sample_matrix)[-3] / sample_size
        per_round_eps.append(np.abs(true_eig - approx_eigval) / n)

    per_size_eps_mean.append(np.mean(per_round_eps))
    per_size_eps_max.append(np.max(per_round_eps))
    per_size_eps_min.append(np.min(per_round_eps))
    per_size_eps_std.append(np.std(per_round_eps))
    per_size_eps_10p.append(np.percentile(per_round_eps, 10))
    per_size_eps_90p.append(np.percentile(per_round_eps, 90))


for i in range(len(errors)):
  if per_size_eps_90p[i] <= per_size_eps_mean[i]:
      print("stats:", per_size_eps_mean[i], \
                      per_size_eps_max[i], \
                      per_size_eps_min[i], \
                      per_size_eps_std[i], \
                      per_size_eps_10p[i], \
                      per_size_eps_90p[i] )


"""
plotting arrays
"""
# errors = tracked_errors[:,3]
# percent_10 = tracked_tenth_percentile[:,3]
# percent_90 = tracked_ninetieth_percentile[:,3]

# log_errors = np.log(errors)
# log_10_p = np.log(percent_10)
# log_90_p = np.log(percent_90)

# for i in range(len(errors)):
#   if percent_90[i] <= errors[i]:
#       print("errors and percentile:", errors[i], percent_90[i])

"""
plotting functions
"""
# import matplotlib.pyplot as plt
# x_axis = list(range(len(tracked_errors[:,3])))
# plt.plot(x_axis, np.log(tracked_errors[:,3]))
# plt.fill_between(x_axis, np.log(tracked_tenth_percentile[:,3]), np.log(tracked_ninetieth_percentile[:,3]), alpha=0.2, color="#069AF3")
# plt.show()