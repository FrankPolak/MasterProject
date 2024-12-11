import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler

X = np.load('Data/preprocessed_methylation.npy', allow_pickle=True)
# X = np.load('Data/preprocessed_RNAseq.npy', allow_pickle=True)
X = X[:, 1:].T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### Evaluate the optimal number of components
n_components_range = [400, 600, 800, 1000, 1200, 1400, 1600, 1800]
explained_variances = []
# [0.7883829295903012, 0.8135236779780183, 0.8349788374171644, 0.867118346893654, 0.8882382988603914, 0.9042179472939202, 0.9175878502942137, 0.9291483639030749]

# Iterate over different numbers of components
for n_components in tqdm(n_components_range):
    spca = SparsePCA(n_components=n_components, tol=1)
    X_transformed = spca.fit_transform(X_scaled)  # Get transformed data
    
    # Reconstruct the data
    X_reconstructed = np.dot(X_transformed, spca.components_)
    
    # Calculate explained variance
    total_variance = np.var(X_scaled, axis=0).sum()  # Total variance of original data
    reconstruction_variance = np.var(X_reconstructed, axis=0).sum()  # Variance explained by components
    explained_variance = reconstruction_variance / total_variance
    explained_variances.append(explained_variance)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(n_components_range, explained_variances, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Sparsity vs Explained Variance')
plt.grid(True)

# 80% Explained Variance Line
if max(explained_variances) > 0.8:
    ind_80 = np.where(np.array(explained_variances) >= 0.8)[0][0]  # First index where explain > 0.8
    plt.plot([0, n_components_range[ind_80]], [explained_variances[ind_80], explained_variances[ind_80]], 'b--')
    plt.plot([n_components_range[ind_80], n_components_range[ind_80]], [0, explained_variances[ind_80]], 'b--')
    plt.text(n_components_range[ind_80], 0, str(n_components_range[ind_80]), color='blue')
    plt.text(0, explained_variances[ind_80], '80%', color='blue')

# 90% Explained Variance Line
if max(explained_variances) > 0.9:
    ind_90 = np.where(np.array(explained_variances) >= 0.9)[0][0]  # First index where explain > 0.9
    plt.plot([0, n_components_range[ind_90]], [explained_variances[ind_90], explained_variances[ind_90]], 'r--')
    plt.plot([n_components_range[ind_90], n_components_range[ind_90]], [0, explained_variances[ind_90]], 'r--')
    plt.text(n_components_range[ind_90], 0, str(n_components_range[ind_90]), color='red')
    plt.text(0, explained_variances[ind_90], '90%', color='red')

plt.show()

### Run Sparse PCA for the selected n_components (1000)
model = SparsePCA(n_components=1000, tol=0.01)
model.fit(X_scaled)

X_reconstructed = model.transform(X_scaled) @ model.components_
total_variance = np.var(X_scaled, axis=0).sum()  # Total variance of original data
reconstruction_variance = np.var(X_reconstructed, axis=0).sum()  # Variance explained by components

print(f"Explained Variance: {reconstruction_variance / total_variance}")
# Explained Variance: 0.8715261291772868

# Obtain the most important features
threshold = 0.2828 # treshold selected based on the resulting number of features (target: 2000)
spca_features = set(np.unique(np.where(np.abs(model.components_) > threshold)[1]))

# Output sparsity and positions of non-zero elements
print("Non-zero positions: ", len(spca_features))
print(spca_features)

# Non-zero positions: 2022
# {0, 1, 2, 3, 4, 5, 6, 7, ..., 2988, 2990, 2993, 2994, 2995, 2997, 2998, 2999}