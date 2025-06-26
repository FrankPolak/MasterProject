import numpy as np
import pandas as pd

# Load the dataset and take absolute values
df_raw = pd.read_csv('../Data/raw/cancer_data_BRCA_RNASeq2GeneNorm-20160128.csv', index_col=0)
df = df_raw.abs()
print(f"{df.shape[0]} features found.")

# Combine index and data into a single array
df = np.column_stack([df.index, df.to_numpy()])

# Identify sparse rows (more than 50% zeros)
sparse_rows = 0
for i, row in enumerate(df):
    zero_count = np.count_nonzero(row == 0)  # Count zeros in the row
    total_elements = len(row)  # Total number of elements in the row
    percentage_zeros = (zero_count / total_elements) * 100  # Calculate percentage of zeros
    if percentage_zeros > 50:  # Check if the row is sparse
        sparse_rows += 1

print(f"{sparse_rows} sparse rows found.")

# Filter out sparse features (keep rows with <= 50% zeros)
rows_to_keep = [np.count_nonzero(row == 0) / len(row) <= 0.5 for row in df]
filtered_df = df[rows_to_keep]
print(f"{filtered_df.shape[0]} features remaining.")

# Analyze variance per feature and select top features by variance
top_var = 3000
row_variances = np.var(filtered_df[:, 1:], axis=1)  # Calculate variances
filtered_df = filtered_df[np.argpartition(-row_variances, top_var)[:top_var]]  # Select top `top_var` features
print(f"{filtered_df.shape[0]} features remaining.")

# Save the preprocessed data
np.save('../Data/preprocessed_RNAseq.npy', filtered_df)

filtered_df_pd = pd.DataFrame(
    filtered_df[:, 1:],  # Feature data
    columns=df_raw.columns,  # Column names from original data
    index=filtered_df[:, 0]  # Gene names as row index
)
filtered_df_pd.to_csv('../Data/preprocessed_RNAseq.csv')
