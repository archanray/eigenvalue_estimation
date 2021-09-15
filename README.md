This is the code repository for our paper on Sublinear Time Eigenvalue Approximation via Random Sampling.

### Required Python packages and dependencies

We experimented in this version and do not expect this to fail in another version of python 3, but still do not guarantee stability in other versions of python packages.

1. Python (3.8.5)
2. Tqdm (4.50.2)
3. Sklearn (0.23.2)
4. Skimage (0.17.2)
5. Numpy (1.19.5)
6. Networkx (2.5)
7. Matplotlib (3.3.2)
8. Pickle (4.0)

### Running instructions

1. To run the approximation do `python main.py`.
2. To change dataset change lines 64 and 66 accordingly in `main.py`.
3. Display codes are in `display_codes.py`.
4. Matlab demo code for a single eigenvalue is in `matlab_code.m`.
5. Figures are in folder `figures`. Please go in to select dataset and then see `errors`.

### Hyperparameter modifications

The four hyperparameters for our code are: 
a) Number of trials (line 63 in code `main.py`);
b) Similarity measure (line 64 in code `main.py`), is only useful for datasets where a similarity measure is required. Otherwise set to `default`;
c) Search rank (line 65 in `main.py`), is used to track the eigenvalues and errors of specified eigenvalues;
d) Dataset name (line 66 in `main.py`).

Please take a look at lines 61-67 in code `main.py` for more details.
