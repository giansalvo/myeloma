# https://gist.github.com/tiagochst/277651ebed998fd3d1952d3fbc376ef2
# https://www.bioconductor.org/packages/release/bioc/vignettes/TCGAbiolinks/inst/doc/casestudy.html
# -----------------------------------------
# TCGAbiolinks vignette Case nb 2
# -----------------------------------------
library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("EDASeq", force = TRUE)

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("genefilter")

query.exp <- GDCquery(
  project = "TCGA-LGG", 
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts",
  barcode = c ("TCGA-WY-A85D-01A-11R-A36H-07", 
               "TCGA-HT-7620-01A-11R-2256-07", 
               "TCGA-DH-5144-01A-01R-1470-07")
)
#  sample.type = c("Primary Tumor")

GDCdownload(
  query = query.exp,
  method = "api",
  directory = "TCGABiolink_study2",
  files.per.chunk = 50
)

lgg.exp <- GDCprepare(
  query = query.exp, 
  save = TRUE, 
  save.filename = "prep.rda",
  directory = "TCGABiolink_study2"
)



dataPrep <- TCGAanalyze_Preprocessing(
  object = lgg.exp, 
  cor.cut = 0.6
)
dataNorm <- TCGAanalyze_Normalization(
  tabDF = dataPrep,
  geneInfo = geneInfoHT,
  method = "gcContent"
)

head(dataNorm)

datFilt <- dataNorm %>% 
  TCGAanalyze_Filtering(method = "varFilter") %>%
  TCGAanalyze_Filtering(method = "filter1") %>%  
  TCGAanalyze_Filtering(method = "filter2",foldChange = 1)

data_Hc2 <- TCGAanalyze_Clustering(
  tabDF = datFilt,
  method = "consensus",
  methodHC = "ward.D2"
) 
# Add  cluster information to Summarized Experiment
colData(lgg.exp)$groupsHC <- paste0("EC",data_Hc2[[4]]$consensusClass)


p <- TCGAanalyze_survival(
  data = colData(lgg.exp),
  clusterCol = "groupsHC",
  main = "TCGA kaplan meier survival plot from consensus cluster",
  legend = "RNA Group",
  height = 10,
  risk.table = T,
  conf.int = F,
  color = c("black","red","blue","green3"),
  filename = "survival_lgg_expression_subtypes.png"
)
plot(p)
