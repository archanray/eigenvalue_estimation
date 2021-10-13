
"""
create/get dataset
"""

import skimage.io
from skimage import feature
import numpy as np
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
    

