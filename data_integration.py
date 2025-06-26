import numpy as np
import pandas as pd
import random
from feature_selection import method_2_NMF

# 1. Import raw data
rna_seq_features = method_2_NMF(hs_file="../Data/hs_RNAseq.npy", features_per_comp=8)
dna_meth_features = method_2_NMF(hs_file="../Data/hs_methylation.npy", features_per_comp=10)

rna_seq_samples = pd.read_csv("../Data/preprocessed_RNAseq.csv", index_col=0)
dna_meth_samples = pd.read_csv("../Data/preprocessed_methylation.csv", index_col=0)

# 2. Keep only selected features
rna_seq_samples = rna_seq_samples.iloc[rna_seq_features,:]
dna_meth_samples = dna_meth_samples.iloc[dna_meth_features,:]

# 3. Keep only common samples
rna_seq_samples.columns = ['-'.join(s.split('-')[:3]) for s in rna_seq_samples.columns]
dna_meth_samples.columns = ['-'.join(s.split('-')[:3]) for s in dna_meth_samples.columns]
common_samples = list(set(rna_seq_samples.columns) & set(dna_meth_samples.columns))

rna_seq_samples = rna_seq_samples.loc[:, common_samples]
dna_meth_samples = dna_meth_samples.loc[:, common_samples]

# 4. Train-test split
random.shuffle(common_samples)
split_index = int(0.6 * len(common_samples)) # 60:40 split

train_samples = common_samples[:split_index]
test_samples = common_samples[split_index:]

rna_seq_train_samples = rna_seq_samples.loc[:,train_samples]
rna_seq_test_samples = rna_seq_samples.loc[:,test_samples]

dna_meth_train_samples = dna_meth_samples.loc[:,train_samples]
dna_meth_test_samples = dna_meth_samples.loc[:,test_samples]

# 5. Export data
rna_seq_train_samples.to_csv('../Data/RNAseq_train.csv')
rna_seq_test_samples.to_csv('../Data/RNAseq_test.csv')

dna_meth_train_samples.to_csv('../Data/DNAMethylation_train.csv')
dna_meth_test_samples.to_csv('../Data/DNAMethylation_test.csv')