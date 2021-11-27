
"""
create/get dataset
"""

import skimage.io
from skimage import feature
import numpy as np
from sklearn.preprocessing import normalize
from display_codes import display_image

def get_data(name):
    if name == "kong":
        imagedrawing = skimage.io.imread('donkeykong.tn768.png')
        display_image(imagedrawing)
        dataset_size = 5000
        edges = imagedrawing
        xy = np.stack(np.where(edges == 0), axis=1)
        n_samples = dataset_size
        xy_sampled_idxs = np.random.randint(low=0, high=xy.shape[0], size=n_samples)
        xy = xy[xy_sampled_idxs, :]
        xy[:,0] = -xy[:,0]
        y_min = np.min(xy[:,0])
        xy[:,0] = xy[:,0]-y_min
        xy = xy.astype(np.float)
        xy[:, 0] = xy[:,0] / np.max(xy[:,0])
        xy[:, 1] = xy[:,1] / np.max(xy[:,1])

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return xy, dataset_size, min_sample_size, max_sample_size

    if name == "asymmetric":
        """
        experiments with asymmetrics 
        """
        dataset_size = 5000
        xy = np.random.random((dataset_size, dataset_size))

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return xy, dataset_size, min_sample_size, max_sample_size

    if name == "binary":
        """
        mimics lower bound code
        """
        dataset_size = 5000
        c = 0.50
        A = np.zeros((dataset_size, dataset_size))
        ind = np.random.choice(range(dataset_size), size=int(dataset_size*c), replace=False)
        A[ind, ind] = -1

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return A, dataset_size, min_sample_size, max_sample_size

    if name == "random_sparse":
        """
        mimics test.m code
        """
        dataset_size = 5000
        A = np.random.random((dataset_size, dataset_size))
        A = A>0.99
        A = A.astype(int)
        A = np.triu(A) + np.triu(A).T

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return A, dataset_size, min_sample_size, max_sample_size

    if name == "block":
        """
        uses a matrix with n/2 x n/2 block of all 1s and rest zeros
        """
        dataset_size = 5000
        A = np.zeros((dataset_size, dataset_size))
        B = np.ones((int(dataset_size/2), int(dataset_size/2)))
        A[0:len(B), 0:len(B)] = B

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return A, dataset_size, min_sample_size, max_sample_size

    if name == "arxiv" or name == "facebook" or name == "erdos":
        """
        dataset arxiv: https://snap.stanford.edu/data/ca-CondMat.html
        dataset facebook: https://snap.stanford.edu/data/ego-Facebook.html
        """
        if name == "arxiv":
            data_file = "./data/CA-CondMat.txt"
        if name == "facebook":
            data_file = "./data/facebook_combined.txt"    
        import networkx as nx
        if name == "erdos":
            from networkx.generators.random_graphs import erdos_renyi_graph
            g = erdos_renyi_graph(5000, p=0.7)
        else:
            g = nx.read_edgelist(data_file,create_using=nx.DiGraph(), nodetype = int)
        A = nx.adjacency_matrix(g)
        A = A.todense()
        if name == "facebook":
            A = A+A.T # symmetrizing as the original dataset is directed

        dataset_size = len(A)
        
        return A, dataset_size

    if name == "multi_block_synthetic":
        from random import sample
        n = 5000
        eps = 0.1

        A = np.ones((n,n))
        num_blocks = round(1/(eps**2))
        sample_block_sizes = round((eps**2)*n)

        R = np.zeros_like(A)
        Z = [-1, 1]

        sample_block = np.ones((sample_block_sizes, sample_block_sizes))

        block_start_row = []
        block_end_row = []
        block_start_col = []
        block_end_col = []
        start_row = 0
        start_col = 0
        for i in range(num_blocks):
            block_start_row.append(start_row)
            block_start_col.append(start_col)
            start_row += sample_block_sizes
            start_col += sample_block_sizes
            block_end_row.append(start_row)
            block_end_col.append(start_col)


        row_id = 0
        for i in range(num_blocks):
            col_id = 0

            for j in range(num_blocks):
                q = int(np.unique(R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]])[-1])

                if q == 0:
                    flag = sample(Z,1)[-1]
                    R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]] \
                                = int(flag)*sample_block

                    R[block_start_row[j]:block_end_row[j], block_start_col[i]:block_end_col[i]] \
                                = flag*sample_block
        A = A+R
        return A, n, 50, 1000

    if name == "synthetic_tester":
        """
        uses a matrix with 1/eps^2 eigvals of size  +-eps*n and 1 eigenvalue of size +-n/2
        """
        print("extracting eigenvectors")
        dataset_size = 2500
        A_p = np.random.random((dataset_size, dataset_size))
        A_p = A_p @ A_p.T
        A_p = A_p / np.max(A_p)
        eigvals, eigvecs = np.linalg.eig(A_p)
        # print(eigvals.shape)
        # mask = np.random.random((dataset_size, dataset_size)) #< 0.2
        # eigvecs = np.where(mask, 0*eigvecs, eigvecs)
        # eigvecs = normalize(eigvecs, axis=0, norm="l2")
        eigvecs[:,0] = (1/np.sqrt(dataset_size))*np.ones(dataset_size)
        
        print("extracting eigenvalues")
        eigvals = (1e-6)*np.ones(dataset_size)
        eps = 0.1
        small_eigs = eps*dataset_size
        large_eig = dataset_size/2
        lens = int(1/(eps**2))
        ones = np.ones(lens)
        all_small_eigs = ones*small_eigs
        eigvals[0] = large_eig
        eigvals[1:lens+1] = all_small_eigs
        mask = np.random.random(dataset_size) < 0.5
        eigvals = np.where(mask, -eigvals, eigvals)
        print(eigvals)
        eigvals_matrix = np.diag(eigvals)

        print("generating final matrix")
        A = (eigvecs @ eigvals_matrix) @ eigvecs.T

        # return A, dataset_size

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return A, dataset_size, min_sample_size, max_sample_size

    if name == "random_equal_signs":
        """
        dataset for tracking frobenius norm error of BSSB-BB
        """
        dataset_size = 2000
        A = np.random.random((dataset_size, dataset_size))
        A = A.T @ A
        w, v = np.linalg.eig(A)
        w = np.array(list(range(2000))).astype(float)
        w = w - 1000
        w[0:500] = w[0:500] - 100.0*np.ones(w[0:500].shape) + np.random.rand(len(w[0:500]))
        w[-500:] = w[-500:] + 100.0*np.ones(w[-500:].shape) + np.random.rand(len(w[-500:]))
        w[500:1501] = np.zeros(w[500:1501].shape) + np.random.rand(len(w[500:1501]))

        # plot eigenvalues
        import matplotlib.pyplot as plt
        plt.plot(np.array(list(range(2000))), w)
        plt.xlabel("eigenvalue indices")
        plt.ylabel("eigenvalues")
        plt.title("eigenvalues of random matrix")
        plt.savefig("./figures/random_equal/eigenvalues/eigvals.pdf")

        w_half = np.lib.scimath.sqrt(w)
        B = v @ np.diag(w_half)

        return B
    

