import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def perform_nmf(input_file, output_file, num_comp):
    """
    Perform NMF decomposition on a given matrix.

    Parameters:
        input_file (str): Path to the input .npy file containing the matrix.
        output_file (str): Path to save the resulting components.
        num_comp (int): Number of components for NMF.

    Returns:
        float: Mean squared error of the reconstruction.
    """
    # Load matrix from the input file
    df = np.load(input_file, allow_pickle=True)
    df = df[:, 1:]
    df = df.T

    # Normalize the data
    scaler = MinMaxScaler()
    df_norm = scaler.fit_transform(df)

    # NMF Hyperparameters
    solver = 'mu'
    max_iter = 500
    tol = 1e-4

    # Perform NMF decomposition
    nmf = NMF(n_components=num_comp, solver=solver, max_iter=max_iter, tol=tol, random_state=42)
    W = nmf.fit_transform(df_norm)
    H = nmf.components_

    # Reconstruct matrix and calculate error
    reconstructed_matrix = np.dot(W, H)
    error = mean_squared_error(df_norm, reconstructed_matrix)

    # Save components to the output file
    np.save(output_file, H)

    print(f"Components: {num_comp}, MSE: {error}")
    return error

if __name__ == "__main__":
    
    error_rna_seq = perform_nmf('Data/preprocessed_RNAseq.npy', 'Data/hs.npy', 250)
    error_dna_meth = perform_nmf('Data/preprocessed_methylation.npy', 'Data/hs_methylation.npy', 200)