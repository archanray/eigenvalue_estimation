
"""
create/get dataset
"""

import skimage.io
from skimage import feature
import numpy as np
from sklearn.preprocessing import normalize
from display_codes import display_image
import matplotlib.pyplot as plt
from random import sample
import os

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
        
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # save figures
        foldername = "figures/matrices/"
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        plt.imshow(A)
        plt.colorbar()
        plt.savefig(foldername+name+"_matrix.pdf")
        plt.clf()

        plt.scatter(range(n), eigvals, alpha=0.3, marker='o', s=2, edgecolors=None)
        plt.savefig(foldername+name+"_eigvals.pdf")
        plt.clf()

        V = np.abs(eigvecs)
        plt.imshow(V)
        plt.colorbar()
        plt.savefig(foldername+name+"_eigvecs.pdf")
        plt.clf()

        SIP = V.T @ V
        plt.imshow(SIP)
        plt.colorbar()
        plt.savefig(foldername+name+"_abs_IP.pdf")
        plt.clf()

        return A, n, int(n/100), int(n/5)

    if name == "multi_block_outer":
        n = 5000
        eps = 0.1

        A = np.ones((n,n))
        num_blocks = round(1/(eps**2))
        sample_block_sizes = round((eps**2)*n)

        R = np.zeros_like(A)
        Z = [-1, 1]

        set_val = np.array([-1,1])

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
                    vec1 = np.random.choice(set_val, size=sample_block_sizes)
                    vec2 = np.random.choice(set_val, size=sample_block_sizes)
                    sample_block = np.outer(vec1, vec2)
                    R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]] \
                                = sample_block

                    R[block_start_row[j]:block_end_row[j], block_start_col[i]:block_end_col[i]] \
                                = sample_block.T
        A = A+R
        
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # save figures
        foldername = "figures/matrices/"
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        plt.imshow(A)
        plt.colorbar()
        plt.savefig(foldername+name+"_matrix.pdf")
        plt.clf()

        plt.scatter(range(n), eigvals, alpha=0.3, marker='o', s=2, edgecolors=None)
        plt.ylim((-250,250))
        plt.savefig(foldername+name+"_eigvals.pdf")
        plt.clf()

        V = np.abs(eigvecs)
        plt.imshow(V)
        plt.colorbar()
        plt.savefig(foldername+name+"_eigvecs.pdf")
        plt.clf()

        SIP = V.T @ V
        plt.imshow(SIP)
        plt.colorbar()
        plt.savefig(foldername+name+"_abs_IP.pdf")
        plt.clf()

        return A, n, int(n/100), int(n/5)

    if name == "synthetic_tester":
        """
        uses a matrix with 1/eps^2 eigvals of size  +-eps*n and 1 eigenvalue of size +-n/2
        """
        n = 5000
        eps = 0.1
        L = list(eps*n*np.ones(100))
        L = np.array([(n/2.0)] + L)
        mask = np.random.random(101) < 0.5
        L = np.where(mask, -1*L, L)
        L = np.diag(L)

        V = np.random.random((n,101))
        num = 0.4
        mask = np.random.random((n,101)) < num
        V = np.where(mask, 0*V, V)
        I = (1/np.sqrt(n))*np.ones(n)
        V[:,0] = I

        Q, R= np.linalg.qr(V)
        A = Q @ L @ Q.T

        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # save figures
        foldername = "figures/matrices/"
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        plt.imshow(A)
        plt.colorbar()
        plt.savefig(foldername+name+"_matrix.pdf")
        plt.clf()

        plt.scatter(range(n), eigvals, alpha=0.3, marker='o', s=2, edgecolors=None)
        plt.savefig(foldername+name+"_eigvals.pdf")
        plt.clf()

        V = np.abs(eigvecs)
        plt.imshow(V)
        plt.colorbar()
        plt.savefig(foldername+name+"_eigvecs.pdf")
        plt.clf()
        
        SIP = V.T @ V
        plt.imshow(SIP)
        plt.colorbar()
        plt.savefig(foldername+name+"_abs_IP.pdf")
        plt.clf()
        return A, n, int(n/100), int(n/5)

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
        plt.plot(np.array(list(range(2000))), w)
        plt.xlabel("eigenvalue indices")
        plt.ylabel("eigenvalues")
        plt.title("eigenvalues of random matrix")
        plt.savefig("./figures/random_equal/eigenvalues/eigvals.pdf")

        w_half = np.lib.scimath.sqrt(w)
        B = v @ np.diag(w_half)

        return B
    

