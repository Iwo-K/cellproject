library(anndata)
library(batchelor)
library(scater)

#' Convert from AnnData to SingleCellExperiment
anndata_to_sce = function(file){

  data = read_h5ad(file)
  sce <- SingleCellExperiment(assays = List(logcounts = as(t(data$X), "CsparseMatrix")),
                              colData=data$obs,
                              rowData=data$var,
                              reducedDims = data$obsm)
  return(sce)
}

library(Seurat)
#' Convert from AnnData to Seurat object
anndata_to_Seurat = function(file){

  sce = anndata_to_sce(file)
  seu = as.Seurat(sce, counts = NULL, data = 'logcounts')
  return(seu)
}
