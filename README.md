This is the code repository for our paper on [Sublinear Time Eigenvalue Approximation via Random Sampling](https://arxiv.org/pdf/2109.07647.pdf).

TODO: 
1. create a package for python using our methods and publish here.

### Required Python libraries and dependencies

We experimented in this version and do not expect this to fail in another version of python 3, but still do not guarantee stability in other versions of python packages.

1. Python (3.8.5)
2. Tqdm (4.50.2)
3. Sklearn (0.23.2)
4. Skimage (0.17.2)
5. Numpy (1.19.5)
6. Networkx (2.5)
7. Matplotlib (3.3.2)
8. Pickle (4.0)
9. idx2numpy

### Running instructions

1. To run the approximation do `python main.py` for only `random uniform sampling`. To compare across methods like `row norm sampling`, `uniform random sampling` or `ratio of non zero elements per row sampling (nnz sampling)` use `compare_unified.py`.
2. To change dataset change lines 64 and 66 accordingly in `main.py` or line 20 in `compare_unified.py`.
3. Display codes are in `display_codes.py`.
4. Matlab demo code for a single eigenvalue is in `matlab_code.m`.
5. Figures are in folder `figures`. Please go in to select dataset and then see `errors`. All unified comparisons are stored in folders named: `XXXX__norm_v_random_v_nnz/errors` where one can replace `XXXX` for dataset name.

### Hyperparameter modifications

The four hyperparameters for our code are: 

* Number of trials (line 63 in code `main.py`) or (line 18 in `compare_unified.py`).
* Similarity measure (line 64 in code `main.py` or line 27 in `compare_unified.py`), is only useful for datasets where a similarity measure is required. Otherwise set to `default` (can ignore in case of `compare_unified.py`).
* Search ranks (line 65 in `main.py` or line 19 of `compare_unified.py`), is used to track the eigenvalues and errors of specified eigenvalues.
* Dataset name (line 66 in `main.py` or line 20 in `compare_unified.py`).

Please take a look at lines 61-67 in code `main.py` for more details.

### Datasets

Most datasets in the paper are synthetic. There are however two real world graph datasets. These are taken from [SNAP](https://snap.stanford.edu/data/).

1. [Facebook](https://snap.stanford.edu/data/ego-Facebook.html) --  [data](https://snap.stanford.edu/data/facebook_combined.txt.gz).
2. [ArXiv COND-MAT](https://snap.stanford.edu/data/ca-CondMat.html) -- [data](https://snap.stanford.edu/data/ca-CondMat.txt.gz).

Please copy these files in a folder named `data` for our code to run without error. Otherwise modify the file `get_dataset.py` to use the path to your dataset. 

We hope this README has exhaustive literature to guide you in our projects. For any questions or queries feel free to reach out to us over email.

### Citation

If you want to cite our paper please check google scholar.
