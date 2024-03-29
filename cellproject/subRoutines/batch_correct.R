run_SeuratCCA = function(infile,
                         anchor.dims = 1:20,
                         integrate.dims = 1:30,
                         batch_key = 'data_type',
                         reference = '10x',
                         k.anchor = 5,
                         k.filter = 200,
                         k.score = 30,
                         use_vargenes = TRUE,
                         outfile = 'corrected_adata.h5ad'){
  print('Running SeuratCCA')

  seu = anndata_to_Seurat(infile)

  seu_list = SplitObject(seu, split.by = batch_key)
  reference = which(names(seu_list) == reference)
  print(paste0('Reference data: ', names(seu_list)[reference]))
  print(reference)

  if (use_vargenes){
    genemeta = seu@assays$originalexp@meta.features
    vargenes = row.names(seu)[genemeta$highly_variable]
  }
  else {
    vargenes = row.names(seu)
  }

  anchors = FindIntegrationAnchors(object.list = seu_list,
                                   dims = anchor.dims,
                                   reference = reference,
                                   anchor.features = vargenes,
                                   k.anchor = k.anchor,
                                   k.filter = k.filter,
                                   k.score = k.score)
  print(anchors)

  combined = IntegrateData(anchorset = anchors,
                           dims = integrate.dims,
                           features.to.integrate = row.names(seu))
  print(combined)

  #Convertin to AnnData and saving (also converted to RsparseMatrix)
  x = combined@assays$integrated@data
  x = AnnData(X = as(t(x), "RsparseMatrix"),
              obs = combined@meta.data,
              var = combined@assays$integrated@meta.features)
  #Reordering to match the input object
  x = x[colnames(seu),rownames(seu)]$copy()
  x$write_h5ad(outfile, compression = 'lzf')
}
