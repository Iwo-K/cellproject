"""Cellproject package helps with asymmetric scRNA-Seq data integration"""

__version__ = '0.1'

from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from umap.umap_ import UMAP

from .utils import *
from .umapfit import *
from .Rbatch_correct import run_SeuratCCA

def project_cells(target,
                  ref,
                  pca_n_components=None,
                  k=15,
                  obs_columns=['leiden'],
                  fit_pca=True,
                  scale_data=False,
                  use_vargenes=True,
                  umap_ref=None,
                  numeric_fun=np.mean,
                  categorical_how='mode',
                  copy=False):
    """
    Project cells onto a reference AnnData. project_cells uses the PCA space
    to identify nearest neighbors between the target and reference data.
    Based on this it propagates categorical (e.g. clusters) and numeric cell annotation
    (from the .obs attribute) to target cells.

    Function by default assumes that reference and target data are not in a common PCA
    space, thus it fits the target expression data into the reference PC space using
    the rotation matrix. To disable this behavior set fit_pca to False.

    If a UMAP reference object is provided (umap_ref) function will try to fit target
    data into the reference UMAP coordinates.

    Caution1: be careful when using cell cycle regression in reference
    data, it may be best to redo scaling and PCA without cell cycle regression
    prior to the projection.

    Caution2: both ref and target must have the same genes and the same order.

    Note: To increase performance implement pynndescent

    Parameters
    ----------
    target
        AnnData object with cells to be projected
    ref
        AnnData object with reference cells
    k
        integer, number of neighbors to consider for label transfer
    obs_columns
        list of strings, column names to be transferred
    fit_pca
        bool, whether target data should be fit into the reference PC space
        (using its varm['PCs'] attribute)
    scale_data
        bool, whether data should be scale, this options required providing
        means and standard deviations values in the ref AnnData .var
        (automatically stored by scanpy if data was scaled
    use_vargenes
        bool, specify whether variable genes should be used (requires a boolean
        column highly-variable in the .var DataFrame of ref AnnData)
    umap_ref
        umap.UMAP object corresponding to the UMAP coordinates in the ref AnnData,
        if provided the function will try to fit the target data.
    numeric_fun
        function which is applied to each numeric vector of nearest neighbors in the
        reference data. Indicates how to summarise the data, defaults to mean.
    categorical_how
        'mode', 'threshold' or 'distribution' - indicates what types of
        label mapping should be used.
    copy
        bool, wheter to modify target in place or return a copy

    Returns
    -------
    If copy set to True, returns the modified target AnnData,
    otherwise modifies in place, ref AnnData  is never modified.
    """

    if copy:
        target = target.copy()
    ref = ref.copy()

    if scale_data:
        if ('mean' not in ref.var.columns) or ('std' not in ref.var.columns):
            raise Exception('Reference data missing .var.mean or .var.std'
                            ', are you sure it is scaled?')
        print('Running a common scaling for target and reference data')
        scaling_mean = ref.var['mean'].values
        scaling_std = ref.var['std'].values
        custom_scale(target,
                     mean=scaling_mean,
                     std=scaling_std,
                     copy=False)

    if fit_pca:
        # Updating target data PCA by transforming data with reference loadings
        print('Fitting target data (.X) into reference PC space')

        target.obsm['X_pca'] = pca_transform(target,
                                       ref,
                                       n_comps=pca_n_components,
                                       use_vargenes=use_vargenes)

    # Updating UMAP in the target data
    if isinstance(umap_ref, UMAP):
        print('Using the provided UMAP reference to fit new data')

        if 'X_pca' not in target.obsm.keys():
            raise Exception('No .obsm["X_pca"] found in target AnnData')

        target.obsm['X_umap'] = quick_umap_proj(target,
                                                umap_ref=umap_ref,
                                                rep_n_components=pca_n_components)
    else:
        print('No valid UMAP object provided, UMAP parameters left unmodified')

    # Calculating neighbors between the target and reference datasets
    cross_dists = pairwise_distances(target.obsm['X_pca'][:, :pca_n_components],
                                     ref.obsm['X_pca'][:, :pca_n_components])
    cross_nn = kneighbor_mask(cross_dists, k=k)
    target.uns['cross_nn'] = csr_matrix(cross_nn)  # storing nn with distances
    cross_nn = cross_nn > 0  # subsequent funtions take boolean nn matrices

    # adj_score is no of neighbors in reference normalised n cells)
    adj_score = np.sum(cross_nn, axis=0) / target.n_obs
    target.uns['adj_score'] = adj_score

    if len(obs_columns) > 0:
        # Get neighbors from reference data (only req. for distribution label transfer)
        if categorical_how == 'distribution':
            if 'distances' in ref.obsp.keys():
                ref_nn = ref.obsp['distances'] > 0
            else:
                raise Exception("Categorical_how set to 'distribution"
                                "but no reference neighbors found"
                                "could not read .obsp['distances']")
        else:
            ref_nn = None

        # Transferring each .obs columns
        for i in obs_columns:
            x = ref.obs[i]
            if x.dtype == 'object' or pd.api.types.is_categorical_dtype(x):
                target.obs['ref_' + i] = assign_label(cross_nn,
                                                      ref_nn,
                                                      ref_labels=x,
                                                      how=categorical_how)
            elif x.dtype == 'int' or x.dtype == 'float':
                target.obs['ref_' + i] = assign_numeric(cross_nn,
                                                        values=x,
                                                        fun=numeric_fun)
            else:
                pass

    return(target) if copy else None


def nnregress(target,
              ref,
              pca_n_components=None,
              regress=['X', 'pca'],
              cross_nn=None,
              weighted=True,
              copy=False):
    """
    Uses k-nn regression to fit cells from target AnnData into PCA and expression
    space of ref AnnData. Updates (or returns modified AnnData copy) the PCA and
    expression values. Needs running project_cells first.

    Parameters
    ----------
    target
        AnnData with cells to be fitted
    ref
        AnnData with reference cells
    pca_n_components
        int, number of PCA components to consider (best to keep consistent with
        project_cells). None will use all components found in the ref.
    regress
        list of attributes to be predicted, can be: 'X', 'raw', 'pca' or any of the matrices
        in the '.layers' attribute
    cross_nn
        dense numpy array or scipy sparse array (csr) with projected cells as rows and
        reference data in columns. Values are distances between target and ref cells.
        If not provided will attempt to retrieve from .uns['cross_nn']
    weighted
        bool, whether to use weighted regression
    copy
        bool, whether to modify target in place or return a copy

    Returns
    -------
    If copy set to True, returns the modified target AnnData,
    otherwise modifies in place, ref AnnData is never modified.
    Expression matrix is returned as sparse if the reference was sparse.
    """

    if copy:
        target = target.copy()

    # Checking and getting required data
    if cross_nn is None:
        if 'cross_nn' in target.uns.keys():
            cross_nn = target.uns['cross_nn']
        else:
            raise Exception('cross_nn argument not provided and target.uns["cross_nn"]'
                            'not found')
    if issparse(cross_nn):
        cross_nn = cross_nn.toarray()

    if 'X_pca' not in ref.obsm.keys():
        raise Exception('No .obsm["X_pca"] found in ref')

    if pca_n_components is None:
        pca_n_components = ref.obsm['X_pca'].shape[1]

    regress = list(set(regress))
    for i in regress:
        if i not in (['X', 'pca', 'raw'] + list(ref.layers.keys())):
            raise Exception('Invalid regress value')
        if i == 'X':
            target.X = calc_nnregress(ref.X, cross_nn, weighted = weighted)
        elif i == 'pca':
            pca = ref.obsm['X_pca']
            target.obsm['X_pca'] = calc_nnregress(pca, cross_nn, weighted=weighted)
            target.varm['PCs'] = ref.varm['PCs'].copy()
        elif i == 'raw':
            target.raw.X[:,:] = calc_nnregress(ref.raw.X, cross_nn, weighted=weighted)
        else:
            target.layers[i] = calc_nnregress(ref.layers[i], cross_nn, weighted=weighted)

    if copy:
        return target
