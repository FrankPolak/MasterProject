import numpy as np
import pandas as pd

rna_seq_data = 'Data/hs.npy'
dna_meth_data = 'Data/hs_methylation.npy'

hs = np.load(dna_meth_data)
# hs.shape
# (200, 3000)

hs = pd.DataFrame(hs).abs()  
hs = hs.T 

f_per_component = int(2000/hs.shape[0])
top_features_2 = {}
components = []

for n in range(f_per_component):
    hs = hs.loc[:, hs.max().sort_values(ascending=False).index]
    
    for col in hs.columns:
        components.append(col)
        component = hs[col]
        feature = component.idxmax()
        top_features_2[feature] = component.max()
        hs = hs.drop(index=feature, axis=0)

top_features_2 = sorted(set(top_features_2.keys()))

print("List of top contributing features:")
print(top_features_2)

# List of top contributing features:
# [0, 1, 2, 3, 4, 5, 6, 7, ..., 2980, 2982, 2989, 2990, 2993, 2995, 2996, 2999]