import numpy as np
import matplotlib.pyplot as plt

def sample_eig(data, s, similarity_measure, scale=False, rankcheck=0):
    """
    input: original matrix
    output: sample eigenvalue
    if using some function to compute elements, this function allows
    you to run the code without instantiating the whole matrix
    """
    n = len(data)
    list_of_available_indices = range(n)
    sample_indices = np.sort(random.sample(list_of_available_indices, s))
    subsample_matrix = similarity_measure(data[sample_indices,:], data[sample_indices,:])
    # useful for only hermitian matrices
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()
  
    min_eig = np.array(all_eig_val_estimates)[rankcheck]

    if scale == False:
        return min_eig
    else:
        return n*min_eig/float(s)
        
# The eigenvalue estimator
def sample_eig_default(data_matrix, s, scale=False, \
                        rankcheck=0, norm=[], nnzA=0, method="uniform random sample", multiplier=1.0):
    """
    input: original matrix
    output: sample eigenvalue
    requires data matrix to be fully instantiated
    """
    n = len(data_matrix)
    list_of_available_indices = range(n)

    sample_indices = np.sort(np.random.choice(list_of_available_indices, \
        size=s, replace=True, p=norm))
    chosen_p = norm[sample_indices]
    
    subsample_matrix = data_matrix[sample_indices][:, sample_indices]

    sqrt_chosen_p = np.sqrt(chosen_p*s)
    D = np.diag(1 / sqrt_chosen_p)
    subsample_matrix = D @ subsample_matrix @ D        

    if "sparsity sampler" in method:
        original_nnzs = np.count_nonzero(subsample_matrix)
        subsample_matrix = subsample_matrix - np.diag(np.diag(subsample_matrix))
        pipj = np.outer(chosen_p, chosen_p)
        mask = (pipj >= 1/(s*multiplier*nnzA)).astype(int) # assuming s \geq tilde{O}(1/epsilon**2)
        subsample_matrix = subsample_matrix*mask
        nnz_subsample_matrix = np.count_nonzero(subsample_matrix)
        try:
            nnz_subsample_matrix = (nnz_subsample_matrix) / float(original_nnzs)
        except:
            nnz_subsample_matrix = 0
    
    # useful for only hermitian matrices
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()

    min_eig = np.array(all_eig_val_estimates)[rankcheck]
    if scale == False or "CUR" in mode:
        if "sparsity sampler" in method:
            return min_eig, nnz_subsample_matrix
        else:
            return min_eig
    else:
        return n*min_eig/float(s)
