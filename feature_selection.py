import numpy as np
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import pandas as pd

def method_1_NMF(hs_file: str, features_per_comp: int=0):
    hs = np.load(hs_file)
    if features_per_comp > 0:
        f_per_component = features_per_comp
    else:
        f_per_component = int(2000/hs.shape[0])

    top_features = []

    for component in hs:
        # Get indices of the top features for each component
        top_indices = component.argsort()[-f_per_component:][::-1]
        top_features.extend(top_indices)

    # Remove duplicates and convert to a sorted list
    top_features = sorted(set(top_features))

    return top_features


def method_2_NMF(hs_file: str, features_per_comp: int=0):
    hs = np.load(hs_file)
    hs = pd.DataFrame(hs).abs()  
    hs = hs.T 

    if features_per_comp > 0:
        f_per_component = features_per_comp
    else:
        f_per_component = int(2000/hs.shape[1])

    top_features = {}

    for n in range(f_per_component):
        hs = hs.loc[:, hs.max().sort_values(ascending=False).index]
        
        for col in hs.columns:
            component = hs[col]
            feature = component.idxmax()
            top_features[feature] = component.max()
            hs = hs.drop(index=feature, axis=0)

    top_features = sorted(set(top_features.keys()))
    return top_features


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


def method_3_LS(file: str, n_neighbors: int, n_features: int, transpose: bool=False, index_col: bool=True):
    X = np.load(file, allow_pickle=True)
    if index_col:
        X = X[:, 1:]
    if transpose:
        X = X.T

    # Compute Laplacian scores for features
    laplacian_scores = laplacian_score(X, n_neighbors)

    # Select indices of the top 1200 features with the lowest Laplacian scores
    top_feature_indices = np.argpartition(laplacian_scores, n_features)[:n_features]

    # Extract the top feature indices
    top_features_LS = list(top_feature_indices)

    return sorted(set(top_features_LS))


def method_4_FILS(file: str, n_neighbors: int, n_features: int, transpose: bool=False, index_col: bool=True, verbose: bool=False):
    X = np.load(file, allow_pickle=True)
    if index_col:
        X = X[:, 1:]
    if transpose:
        X = X.T

    fils_features = []
    remaining_features = list(range(X.shape[1]))
        
    for _ in range(n_features):
        i=1
        scores = laplacian_score(X[:, remaining_features], n_neighbors)

        best_feature_idx = remaining_features[np.argmin(scores)]
        best_feature_idx = remaining_features[best_feature_idx]

        fils_features.append(best_feature_idx)
        remaining_features.remove(best_feature_idx)

        if verbose:
            print(f"{i}/{n_features} selected.")
            i += 1

    print(sorted(fils_features))


def method_5_SPCA(file: str, n_components: int, threshold: int, tol: int=0.01, transpose: bool=False, index_col: bool=True, verbose: bool=False):
    X = np.load(file, allow_pickle=True)
    if index_col:
        X = X[:, 1:]
    if transpose:
        X = X.T

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ### Run Sparse PCA for the selected n_components (1000)
    model = SparsePCA(n_components=n_components, tol=tol)
    model.fit(X_scaled)

    X_reconstructed = model.transform(X_scaled) @ model.components_
    total_variance = np.var(X_scaled, axis=0).sum()  # Total variance of original data
    reconstruction_variance = np.var(X_reconstructed, axis=0).sum()  # Variance explained by components

    if verbose:
        print(f"Explained Variance: {reconstruction_variance / total_variance}")

    spca_features = set(np.unique(np.where(np.abs(model.components_) > threshold)[1]))

    return spca_features


