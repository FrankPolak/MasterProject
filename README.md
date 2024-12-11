# MasterProject

### Index
1. [Downloading Data](#downloading-data)
2. [NMF Optimisation](#nmf-optimisation)
3. [Feature Selection](#feature-selection)
      - [Methods](#methods)
      - [Results](#results)

### Downloading Data
The **Breast invasive carcinoma** database from TCGA was chosen using the code 'BRCA'.
The data was downloaded using the ```curatedTCGAData()``` function.\
The data was filtered to include only primary tumour samples and stored as a MultiAssayExperiment object called ```cancer_data```.\
The ```cancer_data``` object contains data from five experiments, each a SummarizedExperiment object associated with different omics methods:
1. BRCA_miRNASeqGene-20160128:
   - Type: SummarizedExperiment
   - Contains 1046 miRNA features and 499 samples.

2. **BRCA_RNASeq2GeneNorm-20160128:**
   - Type: SummarizedExperiment
   - Contains 20,501 RNA-seq gene features and 782 samples.

3. BRCA_RPPAArray-20160128:
   - Type: SummarizedExperiment
   - Contains 226 protein features from the RPPA assay and 642 samples.

4. BRCA_Methylation_methyl27-20160128:
   - Type: SummarizedExperiment
   - Contains 27,578 features from the 27k methylation array and 272 samples.

5. **BRCA_Methylation_methyl450-20160128:**
   - Type: SummarizedExperiment
   - Contains 485,577 features from the 450k methylation array and 511 samples.
  
The following figure is an UpSet plot showing the intersection of samples in the five experiments.\
![UpSet Plot of TCGA Data](Figures/upSet.png)

```cancer_data``` was exported as an RDS file (data/raw/rnaseqnorm_meth_rppa_mirna_BRCATCGA.rds).\
Datasets 2 and 5 were selected for future model training.

##### Train-Test Split
The data was split using a 60:40 proportion. The training and testing sets of patients were stored as a csv file (data/raw/patients_BRCATCGA.csv).

### NMF Optimisation
1. Normalisation\
   The matrix of expression was normalised to allow for more interpretable and comparable results (MSE).
2. Choosing the Right Solver\
   Coordinate descent (cd) vs Multiplicative update (mu)\
   (all other parameters kept constant)
   
| Method | Reconstruction Error (MSE) | Convergence Speed (seconds) | Stability (variance over 5 runs) |
|--------|-----------------------------|------------------------------|-----------------------------------|
| cd     | 3.468 e-05                 | 49                           | 0.000369                         |
| mu     | 3.602 e-05                 | 33                           | 0.000226                         |

   While the reconstruction error is higher for mu, the convergence speed and increased stability make it the preffered solver. Reconstruction error can later be minimised by adjusting the ```max_iter``` and ```n_components``` parameters of the NMF() function.
   
3. Optimising the Number of Components\
   Different values of ```n_components``` were tested, and the resulting mean squared error (MSE) was plotted. The resulting elbow plots were analysed to determine the optimal number of components to use. This optimal value balances minimising the MSE while avoiding overfitting.\
   ![RNA seq elbow plot](Figures/RNAseq_MSE_elbow_plot.png)\
   **Selected number of components: 250**

   ![DNA Methylation elbow plot](Figures/DNAMethylation_MSE_elbow_plot.png)\
   **Selected number of components: 200**

### Feature Selection
#### Methods:
1.	Top Features Contributing to NMF components.\
   In this method, the top 8 RNA-seq and top 10 DNA Methylation features per component were extracted. Duplicates were removed.
```
For each component: 
   get the top num_top_features features with the highest value
   Append the features to a the top_features list

Sort top_features and remove duplicates
```
2. Top Feature (iterative)\
The feature with the highest contribution was selected and removed. This was repeated until 2000 features were selected.\
```
For n in f_per_component:
   Sort the components based on their max. Value
   For each sorted component:
      Append component index to components list
      Select the top feature
      Append the feature index and its value to the selected_features dictionary 
      Drop the current feature
```
3. Laplacian Score\
   The Laplacian score is a feature selection method that evaluates the relevance of each feature based on its locality-preserving properties. It emphasizes features that maintain the intrinsic structure of the data.\
   To determine the optimal number of neighbors (n_neighbors) for the Laplacian score calculation, a plot of n_neighbors versus the average Laplacian score was analyzed. This helped identify the point at which the score was maximized without overfitting.\
      ![Laplacian Score n_neighbours graph](Figures/LS_feature_selection.png)\
**Selected n_neighbors: 10**
4. Forward Iterative Laplacian Score Algorithm\
   An algorithm that iteratively selects and removes the feature with the highest Laplacian score.\
   [*Skipped due to runtime complexity*]
5. Sparse PCA
   Sparse PCA is a dimensionality reduction technique that balances data compression with feature sparsity. The first step involved determining the optimal number of components (```n_components```) by analysing its influence on the explained variance ratio. A graph was used to illustrate the relationship between the number of components and the percentage of variance explained.\
   ![Sparse PCA analysis](Figures/SPCA_feature_selection.png)\
   From the graph, **n_components = 1000** was selected as it explained over 80% of the variance while maintaining computational efficiency. This choice ensures a balance between retaining significant information and optimising runtime.\
After performing Sparse PCA on both datasets, the explained variances were:
   * DNA Methylation: **87.13%**
   * RNA-seq: **88.41%**

#### Results:   
1. Number of Features Selected
   | Method               | RNA-seq | DNA Methylation |
   |-----------------------|---------|-----------------|
   | Top Features NMF     | 1502    | 1260            |
   | NMF Iterative        | 2000    | 2000            |
   | Laplacian Score      | 2000    | 2000            |
   | Sparse PCA           | 2032    | 2000            |

2. Commonly Selected Features
   ![Venn Feature Selection](Figures/Venn_feature_selection.png)
   | Features Common In: | RNA-seq | DNA Methylation |
   |----------------------|---------|-----------------|
   | 4 sets (all)         | 612     | 431             |
   | 3 sets               | 1018    | 1045            |
   | 2 sets               | 681     | 908             |
   | Single set           | 670     | 585             |

   Based on these results, the final set of features selected for training the deep learning model were the features common in 3 or more sets.
   * RNA-seq: **1630 selected features**
   * DNA Methylation: **1476 selected features**







   







   


  
