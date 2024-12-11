import pandas as pd
import numpy as np

# Load methylation data and annotation data
methylation_data = pd.read_csv('Data/cancer_data_BRCA_Methylation_methyl450-20160128.csv', index_col=0)
annotation = pd.read_csv("Data/450k_annotation.csv")

# Reset index to prepare for merging
methylation_data = methylation_data.reset_index()

# Merge methylation data with annotation data based on probe IDs
merged_data = methylation_data.merge(
    annotation[['Name', 'UCSC_RefGene_Name']],
    left_on='index',
    right_on='Name',
    how='left'
)

# Replace row names with gene names and remove unused columns
merged_data = merged_data.set_index('UCSC_RefGene_Name').drop(columns=['index', 'Name'])

# Remove rows with missing gene names
merged_data = merged_data[~merged_data.index.isna()]

# Use the first gene name before a semicolon as the row index
merged_data.index = merged_data.index.str.split(';').str[0]

# Replace NaN values with 0
merged_data = merged_data.fillna(0)

# Combine index and data into a single array
df = np.column_stack([merged_data.index, merged_data.to_numpy()])

# Identify sparse rows (more than 50% zeros)
sparse_rows = 0
for i, row in enumerate(df):
    zero_count = np.count_nonzero(row == 0)  # Count zeros in the row
    total_elements = len(row)  # Total elements in the row
    percentage_zeros = (zero_count / total_elements) * 100  # Calculate percentage of zeros
    if percentage_zeros > 50:  # Check if the row is sparse
        sparse_rows += 1

print(f"{sparse_rows} sparse rows found.")

# Filter out sparse rows (keep rows with <= 50% zeros)
rows_to_keep = [np.count_nonzero(row == 0) / len(row) <= 0.5 for row in df]
filtered_df = df[rows_to_keep]
print(f"{filtered_df.shape[0]} features remaining.")

# Calculate variances and select top features by variance
top_var = 3000
row_variances = np.var(filtered_df[:, 1:], axis=1)  # Calculate variance for each row
filtered_df = filtered_df[np.argpartition(-row_variances, top_var)[:top_var]]  # Select top `top_var` features
print(f"{filtered_df.shape[0]} features remaining.")

# Save preprocessed data to a numpy file
np.save('Data/preprocessed_methylation.npy', filtered_df)

# Save preprocessed data to a CSV file for easier analysis
filtered_df_pd = pd.DataFrame(
    filtered_df[:, 1:],  # Feature data
    columns=merged_data.columns,  # Column names from original data
    index=filtered_df[:, 0]  # Gene names as row index
)
filtered_df_pd.to_csv('Data/preprocessed_methylation_df.csv')
