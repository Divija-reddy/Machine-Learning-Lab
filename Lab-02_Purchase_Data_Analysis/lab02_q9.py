# Lab Session 02 - A9
# Data Normalization / Scaling

import pandas as pd
import numpy as np

# FUNCTION DEFINITIONS 

def load_thyroid_data(file_path):
    """
    Loads thyroid0387_UCI worksheet from Excel file
    """
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def identify_numeric_attributes(dataframe):
    """
    Identifies numeric attributes that may require normalization
    """
    return dataframe.select_dtypes(include=[np.number]).columns


def min_max_normalization(series):
    """
    Applies Min-Max normalization to a numeric series
    """
    min_val = series.min()
    max_val = series.max()

    if max_val - min_val == 0:
        return series

    return (series - min_val) / (max_val - min_val)


def z_score_normalization(series):
    """
    Applies Z-score normalization to a numeric series
    """
    mean_val = series.mean()
    std_val = series.std()

    if std_val == 0:
        return series

    return (series - mean_val) / std_val


def normalize_data(dataframe):
    """
    Normalizes numeric attributes using appropriate techniques
    """
    normalized_dataframe = dataframe.copy()
    numeric_columns = identify_numeric_attributes(dataframe)

    for column in numeric_columns:
        # Z-score for attributes with wider range / variance
        if normalized_dataframe[column].std() > 1:
            normalized_dataframe[column] = z_score_normalization(
                normalized_dataframe[column]
            )
        else:
            # Min-Max for attributes with limited range
            normalized_dataframe[column] = min_max_normalization(
                normalized_dataframe[column]
            )

    return normalized_dataframe


# MAIN FUNCTION 

def main():
    file_path = "Purchase Data.xlsx"

    # Load dataset
    thyroid_data = load_thyroid_data(file_path)

    # Identify numeric attributes
    numeric_attributes = identify_numeric_attributes(thyroid_data)

    # Normalize data
    normalized_data = normalize_data(thyroid_data)


    print("Numeric Attributes Identified for Normalization:\n")
    print(list(numeric_attributes))

    print("\nSample Data Before Normalization:\n")
    print(thyroid_data[numeric_attributes].head())

    print("\nSample Data After Normalization:\n")
    print(normalized_data[numeric_attributes].head())



if __name__ == "__main__":
    main()
