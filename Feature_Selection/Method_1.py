import numpy as np

rna_seq_data = 'Data/hs.npy'
dna_meth_data = 'Data/hs_methylation.npy'

hs = np.load(dna_meth_data)
# hs.shape
# (200, 3000)

f_per_component = int(2000/hs.shape[0])
top_features_1 = []

for component in hs:
    # Get indices of the top features for each component
    top_indices = component.argsort()[-f_per_component:][::-1]
    top_features_1.extend(top_indices)

# Remove duplicates and convert to a sorted list
top_features_1 = sorted(set(top_features_1))

print("List of top contributing features:")
print(top_features_1)
print(f"Number of features selected: {len(top_features_1)}")

# List of top contributing features:
# [0, 1, 2, 3, 4, 5, 6, 7, ..., 2968, 2969, 2970, 2972, 2974, 2976, 2978, 2989]
# Number of features selected: 1260
