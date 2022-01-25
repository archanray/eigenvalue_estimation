import numpy as np


# The eigenvalue estimator
def sample_eig_default(data_matrix, s, scale=False, rankcheck=0, mode="random_sample", norm=[]):
    """
    input: original matrix
    output: sample eigenvalue
    requires data matrix to be fully instantiated
    """
    n = len(data_matrix)
    list_of_available_indices = range(n)
    if mode == "random_sample":
        norm = np.ones(len(data_matrix)) / n
        sample_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=True, p=norm))
        chosen_p = norm[sample_indices]
        subsample_matrix = data_matrix[sample_indices][:, sample_indices]
        # reweight using pj/s
        sqrt_chosen_p = np.sqrt(chosen_p/s)
        # sqrt_chosen_p = np.ones_like(chosen_p)
        D = np.diag(1 / sqrt_chosen_p)
        subsample_matrix = D @ subsample_matrix @ D
    
    if mode == "norm":
        sample_indices = np.sort(np.random.choice(list_of_available_indices, size=s, replace=True, p=norm))
        chosen_p = norm[sample_indices]
        subsample_matrix = data_matrix[sample_indices][:, sample_indices]
        # compute Ds
        sqrt_chosen_p = np.sqrt(chosen_p*s)
        D = np.diag(sqrt_chosen_p)
        subsample_matrix = D @ subsample_matrix @ D
    
    # useful for only hermitian matrices
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()

    min_eig = np.array(all_eig_val_estimates)[rankcheck]
    if scale == False or "CUR" in mode:
        return min_eig
    else:
        return n*min_eig/float(s)