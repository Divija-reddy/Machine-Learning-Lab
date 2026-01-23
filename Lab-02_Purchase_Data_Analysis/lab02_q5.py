# Lab Session 02 - A5
# Similarity Measures: Jaccard Coefficient & Simple Matching Coefficient

import pandas as pd
import numpy as np

# FUNCTION DEFINITIONS 

def load_thyroid_data(file_path):
    """
    Loads thyroid0387_UCI worksheet from Excel file
    """
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def extract_binary_attributes(dataframe):
    """
    Extracts columns having only binary values (0 and 1)
    """
    binary_columns = []
    for column in dataframe.columns:
        unique_values = dataframe[column].dropna().unique()
        if set(unique_values).issubset({0, 1}):
            binary_columns.append(column)
    return binary_columns


def compute_frequencies(vector1, vector2):
    """
    Computes f11, f10, f01, f00 for two binary vectors
    """
    f11 = f10 = f01 = f00 = 0

    for v1, v2 in zip(vector1, vector2):
        if v1 == 1 and v2 == 1:
            f11 += 1
        elif v1 == 1 and v2 == 0:
            f10 += 1
        elif v1 == 0 and v2 == 1:
            f01 += 1
        elif v1 == 0 and v2 == 0:
            f00 += 1

    return f11, f10, f01, f00


def jaccard_coefficient(f11, f10, f01):
    """
    Calculates Jaccard Coefficient safely
    """
    denominator = f11 + f10 + f01
    if denominator == 0:
        return 0
    return f11 / denominator


def simple_matching_coefficient(f11, f10, f01, f00):
    """
    Calculates Simple Matching Coefficient safely
    """
    total = f11 + f10 + f01 + f00
    if total == 0:
        return 0
    return (f11 + f00) / total


# MAIN FUNCTION 

def main():
    file_path = "Purchase Data.xlsx"

    # Load dataset
    thyroid_data = load_thyroid_data(file_path)

    # Extract binary attributes
    binary_columns = extract_binary_attributes(thyroid_data)

    # Take first two observation vectors
    vector_1 = thyroid_data.loc[0, binary_columns].values
    vector_2 = thyroid_data.loc[1, binary_columns].values

    # Compute frequency counts
    f11, f10, f01, f00 = compute_frequencies(vector_1, vector_2)

    # Compute similarity measures
    jc_value = jaccard_coefficient(f11, f10, f01)
    smc_value = simple_matching_coefficient(f11, f10, f01, f00)


    print("Binary Attributes Considered:")
    print(binary_columns)

    print("\nFrequency Counts:")
    print("f11:", f11)
    print("f10:", f10)
    print("f01:", f01)
    print("f00:", f00)

    print("\nJaccard Coefficient (JC):", jc_value)
    print("Simple Matching Coefficient (SMC):", smc_value)

    print("\nInference:")
    print("JC ignores 0-0 matches and becomes undefined when no positive matches exist.")
    print("SMC considers both matches and mismatches and is more stable in such cases.")



if __name__ == "__main__":
    main()
