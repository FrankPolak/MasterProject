import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

def laplacian_score(X, n_neighbors):
    """
    Computes the Laplacian Score for each feature in the dataset. 
    The Laplacian Score evaluates the relevance of features based on their ability to preserve the locality of the data. 

    Parameters:
    - X (numpy.ndarray): A 2D array of shape (n_samples, n_features), where rows are samples and columns are features.
    - n_neighbors (int): The number of neighbors to use for constructing the similarity graph.

    Returns:
    - laplacian_scores (numpy.ndarray): A 1D array of length n_features, where each entry 
                                        represents the Laplacian Score of the corresponding feature. 
                                        Lower scores indicate more locality-preserving features.
    """
    # similarity graph
    W = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=True).toarray()
    
    # degree matrix
    D = np.diag(W.sum(axis=1))
    
    # Laplacian matrix
    L = D - W
    
    # Calculate the Laplacian score for each feature
    laplacian_scores = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        f = X[:, i]
        f_bar = np.dot(D.sum(axis=0), f) / D.sum()  # Weighted mean of feature i
        numerator = np.dot(f - f_bar, np.dot(L, f - f_bar))
        denominator = np.dot(f - f_bar, np.dot(D, f - f_bar))
        laplacian_scores[i] = numerator / denominator
    
    return laplacian_scores

X = np.load('Data/preprocessed_methylation.npy', allow_pickle=True)
# X = np.load('Data/preprocessed_RNAseq.npy', allow_pickle=True)
X = X[:, 1:]
X = X.T

# Evaluate the optimal number of neighbours per cluster
neighbor_values = [3, 5, 10, 15, 20, 25, 30]
scores = []

for n_neighbors in neighbor_values:
    laplacian_scores = laplacian_score(X, n_neighbors)
    scores.append(np.mean(laplacian_scores)) 

plt.plot(neighbor_values, scores, marker='o')
plt.xlabel("Number of Neighbors")
plt.ylabel("Average Laplacian Score")
plt.title("Effect of n_neighbors on Laplacian Scores")
plt.show()

# Compute Laplacian scores for features
laplacian_scores = laplacian_score(X, 10)

# Map feature indices to their Laplacian scores
score_dict = {index: score for index, score in enumerate(laplacian_scores)}

# Select indices of the top 1200 features with the lowest Laplacian scores
top_feature_indices = np.argpartition(laplacian_scores, 2000)[:2000]

# Extract the top feature indices
top_features_LS = list(top_feature_indices)

print("List of top contributing features:")
print(sorted(set(top_features_LS)))

# List of top contributing features:
# [0, 1, 2, 3, 4, 5, 6, 7, ..., 2989, 2990, 2992, 2993, 2995, 2996, 2998, 2999]