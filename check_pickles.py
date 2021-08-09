import numpy as np
import pickle

dataset_name = "binary"
similarity_measure = "default"

# uncomment to load from pickle file only
with open("pickle_files/"+dataset_name+"_"+similarity_measure+".pkl", "rb") as pickle_file:
    A = pickle.load (pickle_file)
[sample_eigenvalues_scaled, sample_eigenvalues_scaled_std, \
                  tracked_errors, tracked_errors_std] = A

print(sample_eigenvalues_scaled[:,7])
