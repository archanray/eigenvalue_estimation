import numpy as np
from src.get_dataset import get_data
import scipy.sparse as sp
from src.utils import get_distance
from scipy.spatial.distance import cdist

def compute_rbf_kernel(data_matrix):
	dist_mat = cdist(data_matrix,data_matrix,metric="euclidean")
	dist_mat = dist_mat**2 / (2*(0.1)**2)
	dist_mat = -1*dist_mat
	true_mat = np.exp(dist_mat)
	return true_mat

dataset = "avila"
search_ranks = 20 # basically computes the top  k magnitude eigenvalues

if dataset in ["arxiv", "facebook"]:
	true_mat, dataset_size, _, _ = get_data(dataset)
if dataset in ["stsb", "mrpc"]:
	true_mat = np.load("data/"+dataset+"_predicts_0.npy")
	if true_mat.shape[1] == 2:
		true_mat = np.argmax(true_mat, axis=1)
	dataset_size = int(np.sqrt(len(true_mat)))
	true_mat = np.reshape(true_mat, (dataset_size, dataset_size))
	true_mat = (true_mat+true_mat.T) / 2
if dataset in ["avila"]:
	import pandas as pd
	data_matrix = pd.read_csv("data/avila/avila-tr.txt")
	data_matrix = np.array(data_matrix)
	data_matrix = data_matrix[:,0:-1]
	true_mat = compute_rbf_kernel(data_matrix)
	dataset_size = len(true_mat)

sqrt_nnz = np.sqrt(np.count_nonzero(true_mat))

true_mat = true_mat.astype("float")

# get the eigvals necessary
eigvals = sp.linalg.eigsh(true_mat, k=search_ranks, which='LM', return_eigenvectors=False)
nnz_normalized_eigvals = eigvals / sqrt_nnz
n_normalized_eigvals = eigvals / dataset_size

print("*********", dataset, "*************")
print("Eigvals:", eigvals)
print("NNZ normalized eigvals:", nnz_normalized_eigvals)
print("N normalized eigvals:", n_normalized_eigvals)
print("dataset size:", dataset_size)
print("matrix NNZ:", np.count_nonzero(true_mat))
print("***********************************")