import umap
import numpy as np


def quick_umap(adata,
               n_neighbors=None,
               rep_n_components=None,
               min_dist: float = 0.5,
               spread: float = 1.0,
               n_components: int = 2,
               alpha: float = 1.0,
               a=None,
               b=None,
               negative_sample_rate: int = 5,
               init_coords='spectral',
               random_state=0,
               use_rep='X_pca',
               **kwargs):

    '''
    Based on fuction by Xiaonan Wang (smqpp package)
    sc.tl.umap does not save the umap model, which make it's reuse cumbersome
    Most of the default parameters are from scanpy sc.tl.umap function.
    WARNING
    Results are very similar but not identical to the sc.tl.umap function
    Function fits UMAP but does not store any parameters, does not replace any adata
    attributes, the only change is the .obsm['X_umap'] field.

    Input
    -----
    adata: reference anndata object, needs to have 'X_pca' in .obsm and 'neighbors' in
    .uns rep_n_components = 50: specify how many data representation components should
    be used n_neighbors: Number of neighbors to be considered, default: 10
    min_dist: The effective minimum distance between embedded points, default: 0.5
    spread: The effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are. Default: 1.0
    n_components: The number of dimensions of the embedding, default: 2
    alpha: The initial learning rate for the embedding optimization, default: 1.0
    a: More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`. Default: None
    b: More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`. Default: None
    negative_sample_rate: The number of negative edge/1-simplex samples to use per
    positive edge/1-simplex sample in optimizing the low dimensional embedding.
    Default: 5 int_coords: How to initialize the low dimensional embedding.
    Called `init` in the original UMAP. Default: 'spectral'
    random_state: random seed, default: 0
    use_rep: name of the slot in .obsm to use for calculating umap

    **kwargs: other parameters taken in the umap.umap_.UMAP function

    Returns
    -----
    Updates the input adat by adding .obsm['X_umap'] coordinates
    Returns the umap object

    '''
    if 'X_pca' not in adata.obsm_keys():
        raise ValueError('Need to calculate PCA first')

    # Parameters
    if a is None or b is None:
        a, b = umap.umap_.find_ab_params(spread, min_dist)
    if n_neighbors is None:
        try:
            n_neighbors = adata.uns['neighbors']['params']['n_neighbors']
        except Exception as e:
            raise Exception('Either calculae neighbors or provide n_neighbors') from e

    umap_ref = umap.umap_.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        learning_rate=alpha,
        a=a,
        b=b,
        negative_sample_rate=negative_sample_rate,
        init=init_coords,
        random_state=random_state,
        output_metric='euclidean',
        **kwargs
    )

    if rep_n_components is None:
        rep_n_components = adata.obsm[use_rep].shape[1]

    X = adata.obsm[use_rep][:, :rep_n_components]
    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)

    X_umap_fit = umap_ref.fit(X_contiguous)
    X_umap = X_umap_fit.embedding_
    adata.obsm['X_umap'] = X_umap

    return umap_ref


def quick_umap_proj(adata_new,
                    umap_ref,
                    rep_n_components=50,
                    use_rep='X_pca'):
    '''
    Based on fuction by Xiaonan Wang (smqpp package)
    Project new data onto the existing reference data umap based on
    chosen representation (.obsm attribute of AnnData)

    Input
    -----
    adata_new: new anndata object, needs to have the use_rep in .obsm
    umap_ref: umap model as the output of the 'quick_umap' function
    use_rep: name of the key in .obsm attribute to use

    Returns
    -----
    Projected umap coordicates of the new data

    '''

    if use_rep not in adata_new.obsm_keys():
        raise ValueError(f'Need to calculate {use_rep} first')

    X1 = adata_new.obsm[use_rep][:, :rep_n_components]
    X1_contiguous = np.ascontiguousarray(X1, dtype=np.float32)
    X1_umap = umap_ref.transform(X1_contiguous)
    return X1_umap

    # Fitting UMAP is more tricky than PCA as the UMAP object is not stored in the AnnData
    # Solution by Xiaonan addressed this by using a spearate function to run the UMAP and storing the object
    # So this function will rely on providing the UMAP object to allow transformation and then utilise the
    # transformed PCA components to fit the UMAP coordinates.

    # Another problem to solve is cell cycle regression. In most of LK/LSK analyses we are using PCAs computed
    # on cell cycle regressed data. This changes the data structure significantly and makes it incomparable
    # with non-regressed data.
    # For the projected data one could run a similar cell cycle regression, but what if there are very few
    # G2M or S phase cells in that dataset? Running cell cycle regression on combined datasets is not great
    # either as these may be 10x and SS2 so they need to be treated differently.

    # This will be implemented in the future
