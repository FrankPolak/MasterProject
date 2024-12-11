import numpy as np
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

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

fils_features = []
remaining_features = list(range(X.shape[1]))
n_features = 2000
    
for _ in tqdm(range(n_features), desc="Selecting Features", unit="feature"):
    
    scores = laplacian_score(X[:, remaining_features], 10)

    best_feature_idx = remaining_features[np.argmin(scores)]
    best_feature_idx = remaining_features[best_feature_idx]

    fils_features.append(best_feature_idx)
    remaining_features.remove(best_feature_idx)

print(sorted(fils_features))