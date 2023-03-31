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

    # Deliberatel setting groups not in order
    adata.obs['group'] = "group1"
    adata.obs.iloc[3:500, adata.obs.columns == 'group'] = "group2"

    corrected = cp.run_SeuratCCA(adata,
                                      batch_key='group',
                                      reference='group1',
                                      debug = False)


    assert isinstance(corrected, AnnData)

    # cellnames ana group assignments should match
    assert (corrected.obs.index == adata.obs.index).all()
    assert (corrected.obs.group == adata.obs.group).all()

    # The first group should not be corrected and have exactly the same values as original
    c1 = corrected[corrected.obs.group == 'group1',:]
    assert (adata[c1.obs.index, :].X == c1.X).all()

    # The second group should be corrected and thus not match
    c2 = corrected[corrected.obs.group == 'group2',:]
    assert (adata[c2.obs.index, :].X != c2.X).any()

    # Running with the data  ordered by batch should give the same output
    inds1 = np.where(adata.obs.group == 'group1')[0]
    inds2 = np.where(adata.obs.group == 'group2')[0]
    adata_sorted = adata[np.concatenate((inds1, inds2)), :]

    corrected_sorted = cp.run_SeuratCCA(adata_sorted,
                                      batch_key='group',
                                      reference='group1',
                                      debug = False)

    assert (corrected_sorted[corrected.obs.index, :].X != corrected.X).nnz == 0
