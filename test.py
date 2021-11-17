"""
test code for trial
"""

import numpy as np
import random
from tqdm import tqdm
from display_codes import display, display_precomputed_error, display_error_percentile
from get_dataset import get_data
from similarities import sigmoid, tps
import pickle

# labels and other hyperparameters
dataset_name = "block"
similarity_measure_name = "default"
search_rank = -1

# load dataset
dataset, n = get_data(dataset_name)
if similarity_measure_name == "sigmoid":
    similarity_measure = sigmoid
if similarity_measure_name == "tps":
    similarity_measure = tps

if similarity_measure_name == "sigmoid" or similarity_measure_name == "tps":
    true_matrix = similarity_measure(dataset, dataset)
else:
    true_matrix = dataset
true_eig = np.linalg.eigvalsh(true_matrix)[search_rank]

per_size_eps_mean = []
per_size_eps_max = []
per_size_eps_min = []
per_size_eps_std = []
per_size_eps_lowp = []
per_size_eps_highp = []

print("starting approximation")

# perform approximation
for sample_size in tqdm(range(50, 1000, 10)):
    # approx_eigval_list = []
    per_round_eps = []
    for trials in range(50):
        # print(trials)
        samples = np.sort(random.sample(range(n), sample_size))
        sample_matrix = true_matrix[samples][:,samples]
        approx_eigval = n * np.linalg.eigvalsh(sample_matrix)[search_rank] / sample_size
        per_round_eps.append(np.abs(true_eig - approx_eigval) / n)

    per_size_eps_mean.append(np.mean(per_round_eps))
    per_size_eps_max.append(np.max(per_round_eps))
    per_size_eps_min.append(np.min(per_round_eps))
    per_size_eps_std.append(np.std(per_round_eps))
    per_size_eps_lowp.append(np.percentile(per_round_eps, 20))
    per_size_eps_highp.append(np.percentile(per_round_eps, 80))


# for i in range(len(per_size_eps_mean)):
#   if per_size_eps_highp[i] <= per_size_eps_mean[i]:
#       print("stats:", per_size_eps_mean[i], \
#                       per_size_eps_max[i], \
#                       per_size_eps_min[i], \
#                       per_size_eps_std[i], \
#                       per_size_eps_30p[i], \
#                       per_size_eps_70p[i] )


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
# x_axis = np.array(list(range(50, 1000, 10)))
# x_axis = x_axis/n
# plt.plot(np.log(x_axis), np.log(per_size_eps_mean))
# # difference_mat = np.array(per_size_eps_mean)-np.array(per_size_eps_std)
# # difference_mat[difference_mat < 0] = np.array(per_size_eps_30p)[difference_mat < 0]
# # addition_mat = np.array(per_size_eps_mean)+np.array(per_size_eps_std)
# difference_mat = per_size_eps_lowp
# addition_mat = per_size_eps_highp
# plt.fill_between(np.log(x_axis), np.log(difference_mat), \
#                                  np.log(addition_mat), alpha=0.2, color="#069AF3")
# plt.show()
# plt.clf()
# plt.plot(np.log(x_axis), np.log(per_size_eps_mean))
# plt.show()

display_error_percentile(dataset_name, similarity_measure_name, \
                              per_size_eps_mean, 5000, \
                              search_rank, 1000, per_size_eps_lowp, \
                              per_size_eps_highp)
