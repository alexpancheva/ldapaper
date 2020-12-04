library(dplyr)
library(httr)
library(MAST)
library(unixtools)
library(Seurat) 
library(ggplot2)
library(cowplot)
library(future)
library(DoubletFinder)


severe <- readRDS("/data/Alex/COVID19/severeTcellsMacro.rds")

C145Int <- subset(severe,subset=sample=="C145")

DefaultAssay(C145Int) <- "integrated"
DimPlot(object = C145Int , reduction = "umap",label=TRUE,
        group.by = "integrated_snn_res.0.3")


new.cluster.ids <- c("Macrophages 1","Macrophages 2","Macrophages 3","T cells,cluster 1","NK","Macrophages 4","Macrophages 5",
                     "Macrophages 6","T cells, cluster 2","Doublets","Macrophages 7","Macrophages 8")
names(new.cluster.ids) <- levels(C145Int)

C145Int <- RenameIdents(C145Int, new.cluster.ids)
DimPlot(C145Int, reduction = "umap", label = FALSE, pt.size = 0.5) 



DefaultAssay(C145Int) <- "RNA"
FeaturePlot(C145Int, features = c("CD80","CD86","CD40","ITGA4"))

C145Int <- RunUMAP(object = C145Int, reduction = "pca", dims = 1:30)
C145Int <- FindNeighbors(object = C145Int, reduction = "pca", dims = 1:30)
DimPlot(object = C145Int, reduction = "umap",label=TRUE)

C145Int <-  NormalizeData(C145Int, verbose = FALSE)
C145Int <- FindVariableFeatures(C145Int, selection.method = "vst", nfeatures = 2000)
all.genes <- rownames(C145Int)
C145Int <- ScaleData(C145Int, features = all.genes)
C145Int <- RunPCA(C145Int, features = VariableFeatures(object = C145Int))

C145 <- paramSweep_v3(C145Int, PCs = 1:10, sct = FALSE)
sweep.C145 <- summarizeSweep(C145, GT = FALSE)
C145res <- find.pK(sweep.C145)


nExp_poi <- round(0.125*C145Int@assays$RNA@counts@Dim[2])
nExp_poi.adj <- round(nExp_poi*1)

C145Int <- doubletFinder_v3(C145Int, PCs = 1:10, pN = 0.25, pK = 0.3, nExp=nExp_poi, reuse.pANN = FALSE, sct = FALSE)

DefaultAssay(C145Int) <- "integrated"
DimPlot(C145Int, group.by="DF.classifications_0.25_0.3_1640", reduction="umap") +theme(axis.text = element_text(size = 15)) + 
  theme(plot.title = element_text(size = 15))+ theme(legend.text=element_text(size=20))

DimPlot(C145Int, group.by="integrated_snn_res.0.3", reduction="umap",label = TRUE ,label.size = 5) +theme(axis.text = element_text(size = 15)) + 
  theme(plot.title = element_text(size = 15))+ NoLegend()

#save counts and metadata
#write.csv(C145Int@meta.data,"/data/Alex/COVID19/C145forLDAMeta.csv")
#write.csv(C145Int@assays[['RNA']]@counts,"/data/Alex/COVID19/C145forLDACounts.csv")
