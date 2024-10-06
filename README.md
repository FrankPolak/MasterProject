# MasterProject

### Downloading Data
The **Breast invasive carcinoma** database from TCGA was chosen using the code 'BRCA'.
The data was downloaded using the ```curatedTCGAData()``` function.\
The data was filtered to include only primary tumour samples and stored as a MultiAssayExperiment object called ```cancer_data```.\
The ```cancer_data``` object contains data from five experiments, each a SummarizedExperiment object associated with different omics methods:
1. BRCA_miRNASeqGene-20160128:
   - Type: SummarizedExperiment
   - Contains 1046 miRNA features and 499 samples.

2. BRCA_RNASeq2GeneNorm-20160128:
   - Type: SummarizedExperiment
   - Contains 20,501 RNA-seq gene features and 782 samples.

3. BRCA_RPPAArray-20160128:
   - Type: SummarizedExperiment
   - Contains 226 protein features from the RPPA assay and 642 samples.

4. BRCA_Methylation_methyl27-20160128:
   - Type: SummarizedExperiment
   - Contains 27,578 features from the 27k methylation array and 272 samples.

5. BRCA_Methylation_methyl450-20160128:
   - Type: SummarizedExperiment
   - Contains 485,577 features from the 450k methylation array and 511 samples.
  
The following figure is an UpSet plot showing the intersection of samples in the five experiments.\
![UpSet Plot of TCGA Data](Figures/upSet.png)

```cancer_data``` was exported as an RDS file (data/raw/rnaseqnorm_meth_rppa_mirna_BRCATCGA.rds).

### Train-Test Split
The data was split using a 60:40 proportion. The training and testing sets of patients were stored as a csv file (data/raw/patients_BRCATCGA.csv).
  
