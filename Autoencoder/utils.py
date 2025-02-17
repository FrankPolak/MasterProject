import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_data(csv_file_path: str, transpose: bool = True, normalise: bool = True):
    """
    Loads a CSV file, optionally transposes the data, and applies Min-Max normalization.

    Parameters:
    csv_file_path (str): Path to the CSV file.
    transpose (bool, optional): If True, transposes the data. Default is True.
    normalise (bool, optional): If True, applies Min-Max normalization. Default is True.

    Returns:
    pd.DataFrame: Processed data as a DataFrame.
    """
    data = pd.read_csv(csv_file_path, index_col=0)
    
    if transpose:
        data = data.T
    
    if normalise:
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    
    return data