import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse, csr_matrix


def kfilter(x, k=5, keep_values=True):
    """
    Sorts and array and return either a boolean array with
    True set to k smallest values (if keep_values is False) or
    sets all values to 0 and keeps k smallest values.

    Parameters
    ----------
    x
        A numpy array ()
    k
        Integer, number of smallest values to keep
    keep_values
        bool, whether to return boolean (if False) or values (if True)

    Returns
    -------
    numpy array, bool (if keep_values is False), otherwise dtype of x
    """
    inds = np.argsort(x)[range(k)]

    if keep_values:
        out = np.zeros(x.shape, dtype=x.dtype)
        out[inds] = x[inds]
    else:
        out = np.zeros(x.shape, dtype='bool')
        out[inds] = True
    return out


def kneighbor_mask(array, k, axis=1):
    """
    Runs kfilter on a 2 dimensional array, along the specified axis.
    """
    array01 = np.apply_along_axis(kfilter, axis=axis, arr=array, k=k)
    return array01


def custom_scale(adata, mean, std, copy=True):
    '''
    Zero-centered scaling (with unit variance) with defined means and
    standard deviations. Can be used to match scaled gene expression
    values to some refernce dataset.
    This is based on scanpy sc.pp.scale function and works with both
    sparse or dense arrays.

    Note: scanpy stores means and std after scaling in the .var slot

    Parameters
    ----------
    X
        adata - AnnData object (with .X sparse or dense)
    mean
        numpy array with mean gene expression
    std
        numpy array with standard deviations
    copy
        boolean, whether to modify in place or return a copy
    '''
    adata = adata.copy() if copy else adata

    X = adata.X
    if issparse(X):
        X = X.toarray()

    X -= mean
    X /= std

    adata.X = X

    if copy:
        return adata


def get_label_mode(x):
    """
    Convert array to pd.Series and return the mode.
    In case of ties the first value is returned.
    """
    x = pd.Series(x)
    return x.mode()[0]  # what if there ties?


def get_label_by_threshold(x, tr=0.6):
    """
    Get the most prevalent label above a threshold

    Parameters
    ----------
    x
        list/array/pd.Series with labels (strings)
    tr
        float, between 0 and 1. Defines the label fraction threshold.

    Returns
    -------
    Returns the most prevalent value if it occurs more than tr
    fraction. Otherwise returns unassigned.
    """

    x = pd.Series(x)
    x = x.value_counts()
    if any(x >= tr*sum(x)):
        return(x.idxmax())
    else:
        return('unassigned')


def assign_numeric(nn, values, fun=np.mean):
    """
    Assign summarised numeric values based on these values among nearest neighbors
    in the reference datasets.

    Parameters
    ----------
    nn
        bolean numpy array with nearest neighbors between target dataset
        (each cell is a row) and reference dataset (each cell is a column)
    value
        numpy array/pd.Series with numeric values in the reference, needs
        to match the nnmatrix column order
    fun
        function used to summarise the values. Applied to each vector
        of length equal to the number of nearest neighbors

    Returns
    -------
    Numpy array (length equal to rows in nnmatrix) with summarised values
    """
    nobs = nn.shape[0]
    out = np.empty(nobs, dtype=values.dtype)

    for i in range(nobs):
        sub = nn[i, :]
        x = values[sub]
        out[i] = fun(x)
    return out


def count_label_distribution(nn, labels):
    """
    Count normalised label occurences among nearest neighbors.

    Parameters
    ----------
    nn
        a numpy boolean array representing nn adjacency matrix (cells x cells shape)
        each row corresponds to a cell and its k nearest neighbors labelled as True.
    labels
        numpy array/pd.Series with labels, has to correspond to columns in the nn array

    Returns
        pd.DataFrame, rows are cells, columns are labels, values are number of label
        occurences for each label divided by the number of neighbors
    """

    # Detecting k and number of cells (n)
    k = nn[0, :].sum()
    n = nn.shape[0]
    dl = np.empty((n, k), dtype='object')

    for i in range(n):
        dl[i, :] = labels[nn[i, :]]

    dl = pd.DataFrame(dl)
    dl = dl.melt(ignore_index=False, var_name='k', value_name='label')
    out = pd.crosstab(index=dl.index, columns=dl.label)
    out = out/k
    return out

    # Another way: use .value_counts() for each cell and store in an array
    # CAUTION then categories needs to be ordered Otherwise the order changes!

def get_label_by_nndistribution(cross_nn, ref_nn, ref_labels):
    """
    Assign label based on correlation in label distribution among nearest neighbors.
    First we compute the the number of each label among nearest neighbors in the
    reference dataset, then do the same but for nearest neighbors between the
    projected and reference dataset. Finally we compute pairwise correlations
    between and choose the best matching cells in the reference and assign its label.
    In case of ties we take the mode of labels of the best matching cells, and if there
    are ties there we take the first value.

    Parameters
    ----------
    cross_nn
        dense numpy array or scipy sparse array (csr) with projected cells as rows and
        reference data in columns. Values are boolean and indicate if reference cells
        are nearest neighbors of projected cells
    ref_nn
        as in cross_nn but a square matrix indicating nearest neighbors within the
        reference dataset (from rows to columns)
    ref_labels
        numpy array/pd.Series with labels, has to correspond to columns in the
        cross_nn or ref_nn array

    Returns
    -------
    Numpy array with assigned labels. Length equal to the number of projected cells.
    All cells are assigned a label.
    """

    if issparse(cross_nn):
        cross_nn = cross_nn.toarray()
    if issparse(ref_nn):
        ref_nn = ref_nn.toarray()

    ref_ncounts = count_label_distribution(ref_nn, ref_labels)
    target_ncounts = count_label_distribution(cross_nn, ref_labels)
    ref_ncounts = ref_ncounts.loc[:, target_ncounts.columns]

    cordists = pairwise_distances(target_ncounts, ref_ncounts)

    n = target_ncounts.shape[0]
    new_labels = np.empty(n, dtype='object')
    for i in range(n):
        x = cordists[i, :]
        best = np.where(x == x.min())[0]
        if len(best) == 1:
            new_labels[i] = ref_labels[best][0]
        else:
            new_labels[i] = ref_labels[best].mode()[0]

    return new_labels


def assign_label(cross_nn, ref_nn, ref_labels, how='mode'):
    """
    Assign label to projected cells based on the labels among their nearest neighbors
    in the reference dataset.

    Parameters
    ----------
    cross_nn
        dense numpy array or scipy sparse array (csr) with projected cells as rows and
        reference data in columns. Values are boolean and indicate if reference cells
        are nearest neighbors of projected cells
    ref_nn
        as in cross_nn but a square matrix indicating nearest neighbors within the
        reference dataset (from rows to columns)
    ref_labels
        numpy array/pd.Series with labels, has to correspond to columns in the cross_nn
        or ref_nn array
    how
        string, either 'mode', 'threshold' or 'distribution'. Specified the mode
        modified operation, and how the labels will be assigned. See the functions:
        get_label_mode, get_label_by_threshold and get_label_by_nndistribution for
        details.

    Returns
    -------
    Numpy array with assigned labels. Length equal to the number of projected cells.
    If how = 'threshold', some cells may be labelled as 'unassigned'.
    """
    if how == 'mode' or how == 'threshold':
        nobs = cross_nn.shape[0]
        out = np.empty(nobs, dtype='object')

        if how == 'mode':
            fun = get_label_mode

        if how == 'threshold':
            fun = get_label_by_threshold

        for i in range(nobs):
            sub = cross_nn[i, :]
            x = ref_labels[sub]
            out[i] = fun(x)
        return(out)
    elif how == 'distribution':
        return get_label_by_nndistribution(cross_nn, ref_nn, ref_labels)
    else:
        print('Unknown categorical_how, the label is not assigned')


def pca_transform(target,
            ref,
            n_comps=None,
            use_vargenes=True):
    """
    Uses PCA transformation matrix from the reference data (computed if needed)
    to fit the target data into the same PCA space.

    Parameters
    ----------
    target
        AnnData object with cells to be projected
    ref
        AnnData object with reference cells, if the object does not contain
        'X_pca' values in the .obsm attribute, they will be computed
    n_comps
        how many PCA components to use
    use_vargenes
        bool, specify whether variable genes should be used (requires a boolean column
        highly-variable in the .var DataFrame of ref AnnData)

    Returns
    -------
    Numpy array with fitted PCA values of target AnnData (cells as rows, PCs as columns)
    """

    if 'X_pca' not in ref.obsm.keys() and n_comps is None:
        raise Exception('No .obsm["X_pca"] found in ref AnnData and number'
                        'of components unspecified')

    if n_comps is None:
        n_comps = ref.obsm['X_pca'].shape[1]

    if ('PCs' not in ref.varm.keys()):
        print('No PCA rotation (.varm["PCs"]) detected in ref data, running PCA')
        sc.tl.pca(ref, n_comps=n_comps)
    else:
        print('Using existing PCA rotation in .varm["PCs"]')

    if use_vargenes:
        X = target.X[:, ref.var.highly_variable].copy()
        pca_basis = ref.varm['PCs'][ref.var.highly_variable, :n_comps]
        ref_means = ref.X[:, ref.var.highly_variable].mean(axis=0)
    else:
        X = target.X.copy()
        pca_basis = ref.varm['PCs'][:, :n_comps]
        ref_means = ref.X.mean(axis=0, dtype='float64')

    if issparse(ref.X):
        # ref.X is sparse it returns a matrix, need to get a 1d array
        ref_means = ref_means.A1

    if issparse(X):
        X = X.toarray()
    if issparse(pca_basis):
        pca_basis = pca_basis.toarray()

    # By default scanpy zero-centres data before PCA, which causes problems
    # if fitting new data.
    # In cellproject scaling is done to the reference data, so centring is
    # done accordingly. Otherwise PCA is not consistent.
    if ref.uns['pca']['params']['zero_center']:
        X -= ref_means

    # Note: there is some approximation difference when fitting
    # the same data, results are very similar but not identical
    # Converting to float32 to keep consistent with scanpy
    return np.dot(X, pca_basis).astype('float32')


def calc_nnregress(X, cross_nn, weighted=True):
    """
    Predict valus using nearest neighbor regression when supplied with data and
    matrix of nearest neighbors between reference and target.

    Parameters
    ----------
    X
        numpy array or scipy sparse csr array with data used for regression
    cross_nn
        dense numpy array with projected cells as rows and reference data in columns.
        Values are distances between target and ref cells. If not provided will attempt
        to retrieve from .uns['cross_nn']
    weighted
        bool, whether to use weighted regression

    Returns
    -------
    Dense array with regressed values.
    """

    if issparse(X):
        X = X.toarray()

    # Converting to float64 as numpy mean is not alway reproducible with float32
    X = X.astype('float64')

    y = np.empty((cross_nn.shape[0], X.shape[1]), dtype='float32')
    n = cross_nn.shape[0]

    if weighted:
        for i in range(n):
            nns = np.where(cross_nn[i, :] > 0)[0]
            d = cross_nn[i, nns]
            y[i, :] = np.average(X[nns, :], weights=1/d, axis=0).astype('float32')
    else:
        for i in range(n):
            nns = np.where(cross_nn[i, :] > 0)[0]
            d = cross_nn[i, nns]
            y[i, :] = np.average(X[nns, :], axis=0).astype('float32')

    return y

def unify_genes(target, ref, on='index', how='intersection'):
    '''
    Simple function that modifies two AnnData objects in place
    to have the same set of genes.

    There are two mods of operation:
    - how = 'intersection' both ref and target AnnDatas are subset for i in
      range genes existing in both objects
    - how = 'ref' - target AnnData is subset for genes in the ref AnnData,
      genes which are missing are added as all 0 entries.

    The argument on indicates where the gene names/ids are to be found,
    it can either be 'index' (which points to var.index) or
    a name of the column in the .var DataFrame.

    TO BE CONTINUED
    '''
    pass
