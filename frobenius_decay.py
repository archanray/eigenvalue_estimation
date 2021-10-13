import numpy as np
import random
from tqdm import tqdm
from display_codes import frobenius_error_disp
from get_dataset import get_data
import pickle

def sample_frobenius_error(B,s):
    n = len(B)
    list_of_available_indices = range(n)
    sample_indices = np.sort(random.sample(list_of_available_indices, s))
    # subsample_matrix = data_matrix[sample_indices][:, sample_indices]
    S = np.zeros((n,s))
    S[sample_indices, range(s)] = 1.0
    barS = np.sqrt(n/s)*S
    
    BSSB = B.T @ barS @ barS.T @ B
    BB = B.T @ B

    error = np.linalg.norm(BSSB - BB)

    return error

B = get_data("random_equal_signs")
# print(B.shape, type(B))

mean_errors = []
std_errors = []
for s in tqdm(range(50,1000,50)):
    errors = []
    for t in range(5):
        errors.append(sample_frobenius_error(B, s))
    
    mean_errors.append(np.mean(errors))
    std_errors.append(np.std(errors))


### plotter
frobenius_error_disp(np.array(mean_errors),\
            np.array(std_errors),\
            "random_equal", 50, 1000, 50, 2000)
