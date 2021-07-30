'''Cellproject testing'''

import numpy as np
import scanpy as sc
import cellproject as cp
from anndata import AnnData
from scipy.sparse import csr_matrix
import pytest


def scale_comb_and_separate(adata):

    adata1 = adata.copy()
    sc.pp.scale(adata1)
    adata2 = adata.copy()

    adata2 = cp.custom_scale(adata2, mean=adata1.var['mean'].values,
                    std=adata1.var['std'].values)
    return(adata1, adata2)


@pytest.fixture
def paul15_proc():

    adata = sc.datasets.paul15()
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000)
    sc.pp.log1p(adata)
    adata.var['highly_variable'] = True
    return(adata)


@pytest.fixture
def paul15_reference(paul15_proc):

    adata = paul15_proc.copy()
    adata.var['highly_variable'] = True
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=20)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    umapref = cp.quick_umap(adata)
    return(adata, umapref)

def test_custom_scale(paul15_proc):

    adata = paul15_proc.copy()

    out = scale_comb_and_separate(adata)
    assert np.array_equal(out[0].X, out[1].X)

    # Repeating for sparse matrix
    adata.X = csr_matrix(adata.X)
    out = scale_comb_and_separate(adata)
    assert np.array_equal(out[0].X, out[1].X)


def test_project_cells(paul15_proc, paul15_reference):
    # Projecting the same data onto itself should produce consistent results

    adata = paul15_proc.copy()
    reference = paul15_reference[0].copy()
    refumap = paul15_reference[1]

    cp.project_cells(adata,
                     reference,
                     scale_data=True,
                     k=10,
                     obs_columns=['leiden', 'paul15_clusters'],
                     umap_ref=refumap)

    # Checking pca fit
    assert np.allclose(adata.obsm['X_pca'],
                       reference.obsm['X_pca'],
                       rtol=1e-03, atol=1e-05)

    # Checking label transfer
    leiden_correct = (adata.obs.ref_leiden == reference.obs.leiden).sum()
    assert (leiden_correct / reference.n_obs) > 0.95

    # Checking UMAP fit
    # Due to PCA inaccuracy UMAP coordinates are noisy but we check correlation
    u1_r = np.corrcoef(adata.obsm['X_umap'][:, 0], reference.obsm['X_umap'][:, 0])[0, 1]
    u2_r = np.corrcoef(adata.obsm['X_umap'][:, 1], reference.obsm['X_umap'][:, 1])[0, 1]
    assert u1_r > 0.975 and u2_r > 0.975


def test_project_cells_distribution(paul15_proc, paul15_reference):

    adata = paul15_proc.copy()
    reference = paul15_reference[0].copy()
    refumap = paul15_reference[1]

    cp.project_cells(adata,
                     reference,
                     scale_data=True,
                     k=10,
                     obs_columns=['leiden', 'paul15_clusters'],
                     umap_ref=refumap,
                     categorical_how='distribution')

    # Checking label transfer
    leiden_correct = (adata.obs.ref_leiden == reference.obs.leiden).sum()
    assert (leiden_correct / reference.n_obs) > 0.95


def test_nnregress(paul15_proc, paul15_reference):

    adata = paul15_proc.copy()
    reference = paul15_reference[0].copy()
    refumap = paul15_reference[1]

    cp.project_cells(adata,
                     reference,
                     scale_data=True,
                     k=10,
                     obs_columns=['leiden'],
                     umap_ref=refumap)

    cp.nnregress(adata,
                 reference,
                 regress=['X', 'pca'])

    del adata.obsm['X_umap']
    adata.obs = adata.obs.drop('ref_leiden', axis=1)

    cp.project_cells(adata,
                     reference,
                     fit_pca=False,
                     scale_data=False,
                     k=10,
                     obs_columns=['leiden'],
                     umap_ref=refumap)

    # Checking label transfer after nnregression
    leiden_correct = (adata.obs.ref_leiden == reference.obs.leiden).sum()
    assert (leiden_correct / reference.n_obs) > 0.95

    # Checking UMAP fit aftern nnregression
    # Due to PCA inaccuracy UMAP coordinates are noisy but we check correlation
    u1_r = np.corrcoef(adata.obsm['X_umap'][:, 0], reference.obsm['X_umap'][:, 0])[0, 1]
    u2_r = np.corrcoef(adata.obsm['X_umap'][:, 1], reference.obsm['X_umap'][:, 1])[0, 1]
    assert u1_r > 0.975 and u2_r > 0.975


def test_run_seuratCCA(paul15_proc):
    '''Just bare testing if the code works. Need to figure testing if the results are correct'''

    adata = paul15_proc.copy()

    adata1 = adata[1:500, :].copy()
    adata2 = adata[1000:1500, :].copy()

    comb = adata1.concatenate(adata2, batch_key='batch', batch_categories=['batchA', 'batchB'])

    corrected = cp.run_SeuratCCA(comb,
                                      batch_key='batch',
                                      reference='batchA',
                                      debug = False)

    assert isinstance(corrected, AnnData)
