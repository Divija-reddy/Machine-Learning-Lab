# Lab Session 02 - A7
# Heatmap Visualization of Similarity Measures

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# FUNCTION DEFINITIONS 
def load_thyroid_data(file_path):
    """
    Loads thyroid0387_UCI worksheet from Excel file
    """
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def extract_binary_attributes(dataframe):
    """
    Extracts columns with binary values (0/1)
    """
    binary_columns = []
    for column in dataframe.columns:
        unique_values = dataframe[column].dropna().unique()
        if set(unique_values).issubset({0, 1}):
            binary_columns.append(column)
    return binary_columns


def compute_frequencies(vector1, vector2):
    """
    Computes f11, f10, f01, f00
    """
    f11 = f10 = f01 = f00 = 0
    for v1, v2 in zip(vector1, vector2):
        if v1 == 1 and v2 == 1:
            f11 += 1
        elif v1 == 1 and v2 == 0:
            f10 += 1
        elif v1 == 0 and v2 == 1:
            f01 += 1
        else:
            f00 += 1
    return f11, f10, f01, f00


def jaccard_coefficient(f11, f10, f01):
    """
    Computes Jaccard Coefficient safely
    """
    denom = f11 + f10 + f01
    return 0 if denom == 0 else f11 / denom


def simple_matching_coefficient(f11, f10, f01, f00):
    """
    Computes Simple Matching Coefficient
    """
    total = f11 + f10 + f01 + f00
    return 0 if total == 0 else (f11 + f00) / total


def cosine_similarity(vector_a, vector_b):
    """
    Computes Cosine Similarity safely
    """
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(vector_a, vector_b) / (norm_a * norm_b)


def compute_similarity_matrices(dataframe, binary_cols):
    """
    Computes JC, SMC and COS similarity matrices for first 20 observations
    """
    binary_data = dataframe[binary_cols].iloc[:20].values
    numeric_data = dataframe.select_dtypes(include=[np.number]).iloc[:20].values

    size = 20
    jc_matrix = np.zeros((size, size))
    smc_matrix = np.zeros((size, size))
    cos_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            f11, f10, f01, f00 = compute_frequencies(binary_data[i], binary_data[j])
            jc_matrix[i][j] = jaccard_coefficient(f11, f10, f01)
            smc_matrix[i][j] = simple_matching_coefficient(f11, f10, f01, f00)
            cos_matrix[i][j] = cosine_similarity(numeric_data[i], numeric_data[j])

    return jc_matrix, smc_matrix, cos_matrix


def plot_heatmap(matrix, title):
    """
    Plots heatmap for similarity matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="coolwarm")
    plt.title(title)
    plt.xlabel("Observation Index")
    plt.ylabel("Observation Index")
    plt.show()



def main():
    file_path = "Purchase Data.xlsx"

    # Load dataset
    thyroid_data = load_thyroid_data(file_path)

    # Extract binary attributes
    binary_columns = extract_binary_attributes(thyroid_data)

    # Compute similarity matrices
    jc_matrix, smc_matrix, cos_matrix = compute_similarity_matrices(
        thyroid_data, binary_columns
    )

    # PLOTS 

    plot_heatmap(jc_matrix, "Jaccard Coefficient Heatmap (First 20 Observations)")
    plot_heatmap(smc_matrix, "Simple Matching Coefficient Heatmap (First 20 Observations)")
    plot_heatmap(cos_matrix, "Cosine Similarity Heatmap (First 20 Observations)")


# EXECUTION 

if __name__ == "__main__":
    main()
