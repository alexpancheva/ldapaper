library(dplyr)
library(httr)
library(MAST)
library(unixtools)
library(Seurat) 
library(ggplot2)
library(cowplot)
library(future)

dirData = "/data/Alex/COVID19/"

BALF.combined <- readRDS(paste(dirData,"BALF.combined.rds",sep=""))
set.seed(42)
BALF.combined <- FindNeighbors(object = BALF.combined, reduction = "pca", dims = 1:30)
BALF.combined <- FindClusters(BALF.combined, resolution = 0.5)
BALF.combined <- RunUMAP(object = BALF.combined, reduction = "pca", dims = 1:30)


DefaultAssay(BALF.combined) <- "integrated"
png("integrated_group_0.5.png", width = 3000, height = 1000, res=350)
DimPlot(object = BALF.combined, reduction = "umap",split.by='group',label=TRUE,label.size = 5) +theme(axis.text = element_text(size = 15)) + 
        theme(plot.title = element_text(size = 15))+ NoLegend()
dev.off()


DefaultAssay(BALF.combined) <- "RNA"
clusterMarkers <- FindAllMarkers(object = BALF.combined, only.pos = TRUE,test.use="MAST")
top30 <- clusterMarkers %>% group_by(cluster) %>% top_n(n = 30, wt = avg_logFC)
top10 <- clusterMarkers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)
write.csv(clusterMarkers, paste(dirData,"all_samples_markers.csv",sep=""))


#subset clusters
BALF.combined.subset <- subset(BALF.combined, idents=c("9","10","18","11","14"), invert=TRUE)
set.seed(42)
DefaultAssay(BALF.combined.subset) <- "integrated"
BALF.combined.subset <- RunUMAP(object = BALF.combined.subset, reduction = "pca", dims = 1:30)
BALF.combined.subset <- FindNeighbors(object = BALF.combined.subset, reduction = "pca", dims = 1:30)
BALF.combined.subset <- FindClusters(BALF.combined.subset, resolution = 0.2)
DimPlot(object = BALF.combined.subset, reduction = "umap",split.by='group',label=TRUE)

#####################
#using only the subset

DefaultAssay(BALF.combined.subset) <- "integrated"
BALF.combined.subset <- FindClusters(BALF.combined.subset, resolution = c(0.3))
DimPlot(object = BALF.combined.subset, reduction = "umap",split.by='group',label=TRUE,group.by = "integrated_snn_res.0.3", label.size = 5) +theme(axis.text = element_text(size = 15)) + 
        theme(plot.title = element_text(size = 15))+ NoLegend()


DefaultAssay(BALF.combined.subset) <- "RNA"
FeaturePlot(BALF.combined.subset, features = c("CD14"),split.by="group")
FeaturePlot(BALF.combined.subset, features=c("CD3D","CD68"), order=TRUE)

DefaultAssay(BALF.combined.subset) <- "RNA"
clusterMarkers <- FindAllMarkers(object = BALF.combined.subset, only.pos = TRUE,test.use="MAST")
top30 <- clusterMarkers %>% group_by(cluster) %>% top_n(n = 30, wt = avg_logFC)
top10 <- clusterMarkers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)
write.csv(clusterMarkers, paste(dirData,"subset_markers03.csv",sep=""))


tcells <- subset(BALF.combined.subset, idents=c(3,4,8,9),invert=FALSE)
DefaultAssay(tcells) <- "integrated"
tcells <- FindNeighbors(object = tcells, reduction = "pca", dims = 1:30)
tcells <- FindClusters(tcells, resolution = c(0.5))
set.seed(42)
tcells <- RunUMAP(object = tcells, reduction = "pca", dims = 1:30)
DimPlot(object = tcells , reduction = "umap",label=TRUE,
        group.by = "integrated_snn_res.0.3",split.by = "group")

DefaultAssay(tcells) <- "RNA"
FeaturePlot(tcells, features = c("FCN1"),split.by = 'group')
FeaturePlot(tcells, features = c("CD68","FCGR3B"))

DefaultAssay(tcells) <- "RNA"
clusterMarkersTcells <- FindAllMarkers(object = tcells, only.pos = TRUE,test.use="MAST")
top10 <- clusterMarkersTcells %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)


####################
macrophages <- subset(BALF.combined.subset,idents=c(3,4,8,9), invert=TRUE)

DefaultAssay(macrophages) <- "integrated"
macrophages <- FindNeighbors(object = macrophages, reduction = "pca", dims = 1:30)
macrophages <- FindClusters(macrophages, resolution = c(0.5))
set.seed(42)
macrophages <- RunUMAP(object = macrophages, reduction = "pca", dims = 1:30)
DimPlot(object = macrophages , reduction = "umap",label=TRUE,group.by = "integrated_snn_res.0.5")


DefaultAssay(macrophages) <- "RNA"
FeaturePlot(macrophages, features = c("FCN1","SPP1","FABP4"))
FeaturePlot(macrophages, features = c("IGHG4","CD3D"))

DefaultAssay(macrophages) <- "RNA"
clusterMarkersMacro <- FindAllMarkers(object = macrophages, only.pos = TRUE,test.use="MAST")
top10 <- clusterMarkersMacro %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)




#subset severe + split by patient 
severe <- subset(BALF.combined.subset, subset = group == "Severe COVID-19")
DefaultAssay(severe) <- "integrated"
DimPlot(object = severe , reduction = "umap",label=TRUE,
        group.by = "integrated_snn_res.0.3",split.by = "sample",label.size = 3) +theme(axis.text = element_text(size = 15)) + 
        theme(plot.title = element_text(size = 15))+ NoLegend()

DefaultAssay(severe) <- "RNA"
FeaturePlot(severe, features = c("KIT","TPSB2"))
severe <- subset(severe,subset =FCGR3B>1, invert=TRUE) 


healhty <- subset(BALF.combined.subset, subset = group == "Healthy")
DefaultAssay(healhty) <- "integrated"
DimPlot(object = healhty , reduction = "umap",label=TRUE,
        group.by = "integrated_snn_res.0.3",split.by = 'sample')

saveRDS(severe, paste(dirData,"severeTcellsMacro.rds",sep=""))



