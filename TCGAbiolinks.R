knitr::opts_chunk$set(dpi = 300)
knitr::opts_chunk$set(cache=FALSE)

install.packages("TCGAbiolinks")
library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)
library(DT)

library(SummarizedExperiment)
library(TCGAbiolinks)

version
packageVersion("TCGAbiolinks")
##################################################################
# FROM ARTICLE FROM LUKE
query_mm_fpkm <- GDCquery ( project = "MMRF-COMMPASS" ,
                                data.category = "Transcriptome Profiling" ,
                                data.type = "Gene Expression Quantification" ,
                                workflow.type ="STAR - Counts" ,
                                barcode = c ("MMRF_2473", "MMRF_2111", "MMRF_2270",
                                             "MMRF_2238", "MMRF_1080", "MMRF_2253",
                                             "MMRF_2119", "MMRF_2468", "MMRF_1201",
                                             "MMRF_2821", "MMRF_1957", "MMRF_1678"))


GDCdownload(
  query = query_mm_fpkm,
  method = "api",
  directory = "MMRF_example",
  files.per.chunk = 50
)

prep <- GDCprepare(query = query_mm_fpkm, 
                   save = TRUE, 
                   save.filename = "prep.rda",
                   directory = "MMRF_example")

MMRF_prepare_SurvivalKM(prep)


#################################################################

query.exp <- GDCquery(project = "TCGA-BRCA", 
                      legacy = TRUE,
                      data.category = "Gene expression",
                      data.type = "Gene expression quantification",
                      platform = "Illumina HiSeq", 
                      file.type = "results",
                      experimental.strategy = "RNA-Seq",
                      sample.type = c("Primary Tumor","Solid Tissue Normal"))
GDCdownload(query.exp)
brca.exp <- GDCprepare(query = query.exp, save = TRUE, save.filename = "brcaExp.rda")

# get subtype information
dataSubt <- TCGAquery_subtype(tumor = "BRCA")

# get clinical data
dataClin <- GDCquery_clinic(project = "TCGA-BRCA","clinical") 

# Which samples are Primary Tumor
dataSmTP <- TCGAquery_SampleTypes(getResults(query.exp,cols="cases"),"TP") 
# which samples are solid tissue normal
dataSmNT <- TCGAquery_SampleTypes(getResults(query.exp,cols="cases"),"NT")

########################################################

query <- GDCquery(project = "TCGA-ACC",
                  data.category = "Copy Number Variation",
                  data.type = "Copy Number Segment")
## Not run: 
query <- GDCquery(project = "TARGET-AML",
                  data.category = "Transcriptome Profiling",
                  data.type = "miRNA Expression Quantification",
                  workflow.type = "BCGSC miRNA Profiling",
                  barcode = c("TARGET-20-PARUDL-03A-01R","TARGET-20-PASRRB-03A-01R"))
query <- GDCquery(project = "TARGET-AML",
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification",
                  workflow.type = "HTSeq - Counts",
                  barcode = c("TARGET-20-PADZCG-04A-01R","TARGET-20-PARJCR-09A-01R"))
query <- GDCquery(project = "TCGA-ACC",
                  data.category =  "Copy Number Variation",
                  data.type = "Masked Copy Number Segment",
                  sample.type = c("Primary Tumor"))
query.met <- GDCquery(project = c("TCGA-GBM","TCGA-LGG"),
                      legacy = TRUE,
                      data.category = "DNA methylation",
                      platform = "Illumina Human Methylation 450")
query <- GDCquery(project = "TCGA-ACC",
                  data.category =  "Copy number variation",
                  legacy = TRUE,
                  file.type = "hg19.seg",
                  barcode = c("TCGA-OR-A5LR-01A-11D-A29H-01"))

## End(Not run)

###################################################################
#https://rdrr.io/bioc/TCGAbiolinks/f/vignettes/mutation.Rmd

maf <- GDCquery_Maf("CHOL", pipelines = "muse")

maf <- chol_maf@data

# Only first 50 to make render faster
datatable(maf[1:20,],
          filter = 'top',
          options = list(scrollX = TRUE, keys = TRUE, pageLength = 5), 
          rownames = FALSE)
###################################################################
# https://rdrr.io/bioc/TCGAbiolinks/f/vignettes/analysis.Rmd
library(SummarizedExperiment)
# You can define a list of samples to query and download providing relative TCGA barcodes.
listSamples <- c("TCGA-E9-A1NG-11A-52R-A14M-07","TCGA-BH-A1FC-11A-32R-A13Q-07",
                 "TCGA-A7-A13G-11A-51R-A13Q-07","TCGA-BH-A0DK-11A-13R-A089-07",
                 "TCGA-E9-A1RH-11A-34R-A169-07","TCGA-BH-A0AU-01A-11R-A12P-07",
                 "TCGA-C8-A1HJ-01A-11R-A13Q-07","TCGA-A7-A13D-01A-13R-A12P-07",
                 "TCGA-A2-A0CV-01A-31R-A115-07","TCGA-AQ-A0Y5-01A-11R-A14M-07")

# Query platform Illumina HiSeq with a list of barcode 
query <- GDCquery(project = "TCGA-BRCA", 
                  data.category = "Gene expression",
                  data.type = "Gene expression quantification",
                  experimental.strategy = "RNA-Seq",
                  platform = "Illumina HiSeq",
                  file.type = "results",
                  barcode = listSamples, 
                  legacy = TRUE)

# Download a list of barcodes with platform IlluminaHiSeq_RNASeqV2
GDCdownload(query)

# Prepare expression matrix with geneID in the rows and samples (barcode) in the columns
# rsem.genes.results as values
BRCARnaseqSE <- GDCprepare(query)

BRCAMatrix <- assay(BRCARnaseqSE,"raw_count") # or BRCAMatrix <- assay(BRCARnaseqSE,"raw_count")

# For gene expression if you need to see a boxplot correlation and AAIC plot to define outliers you can run
BRCARnaseq_CorOutliers <- TCGAanalyze_Preprocessing(BRCARnaseqSE)

library(TCGAbiolinks)
dataGE <- dataBRCA[sample(rownames(dataBRCA),10),sample(colnames(dataBRCA),7)]

knitr::kable(dataGE[1:10,2:3], digits = 2, 
             caption = "Example of a matrix of gene expression (10 genes in rows and 2 samples in columns)",
             row.names = TRUE)

#######################################################
## Not run:
# Download clinical data from XML
query <- GDCquery(project = "TCGA-COAD", data.category = "Clinical")
GDCdownload(query, files.per.chunk = 200)
query <- GDCquery(
  project = "TARGET-AML",
  data.category = "Transcriptome Profiling",
  data.type = "miRNA Expression Quantification",
  workflow.type = "BCGSC miRNA Profiling",
  barcode = c("TARGET-20-PARUDL-03A-01R", "TARGET-20-PASRRB-03A-01R")
)
# data will be saved in:
# example_data_dir/TARGET-AML/harmonized/Transcriptome_Profiling/miRNA_Expression_Quantification
GDCdownload(query, method = "client", directory = "example_data_dir")
query_acc_gbm <- GDCquery(
  project = c("TCGA-ACC", "TCGA-GBM"),
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)
GDCdownload(
  query = query_acc_gbm,
  method = "api",
  directory = "example",
  files.per.chunk = 50
)
## End(Not run)
