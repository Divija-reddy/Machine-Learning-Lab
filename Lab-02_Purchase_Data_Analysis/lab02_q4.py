# Lab Session 02 - A4
# Data Exploration - Thyroid Dataset

import pandas as pd
import numpy as np

# FUNCTION DEFINITIONS 

def load_thyroid_data(file_path):
    """
    Loads thyroid0387_UCI worksheet from Excel file
    """
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def identify_attribute_types(dataframe):
    """
    Identifies data type of each attribute
    """
    return dataframe.dtypes


def identify_missing_values(dataframe):
    """
    Counts missing values in each column
    """
    return dataframe.isnull().sum()


def numeric_data_range(dataframe):
    """
    Computes min and max for numeric columns
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number])
    return numeric_cols.min(), numeric_cols.max()


def calculate_mean_variance(dataframe):
    """
    Calculates mean and variance for numeric attributes
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number])
    mean_values = numeric_cols.mean()
    variance_values = numeric_cols.var()
    return mean_values, variance_values


def detect_outliers_iqr(dataframe):
    """
    Detects outliers using IQR method for numeric attributes
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number])
    outlier_summary = {}

    for column in numeric_cols.columns:
        Q1 = numeric_cols[column].quantile(0.25)
        Q3 = numeric_cols[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = numeric_cols[
            (numeric_cols[column] < lower_bound) | 
            (numeric_cols[column] > upper_bound)
        ]

        outlier_summary[column] = len(outliers)

    return outlier_summary


def identify_categorical_attributes(dataframe):
    """
    Identifies categorical attributes for encoding
    """
    categorical_cols = dataframe.select_dtypes(include=["object"])
    return categorical_cols.columns


#  MAIN FUNCTION 

def main():
    file_path = "Purchase Data.xlsx"

    # Load thyroid dataset
    thyroid_data = load_thyroid_data(file_path)

    # Attribute type identification
    attribute_types = identify_attribute_types(thyroid_data)

    # Missing values
    missing_values = identify_missing_values(thyroid_data)

    # Numeric data range
    min_values, max_values = numeric_data_range(thyroid_data)

    # Mean and variance
    mean_values, variance_values = calculate_mean_variance(thyroid_data)

    # Outlier detection
    outlier_info = detect_outliers_iqr(thyroid_data)

    # Categorical attributes
    categorical_attributes = identify_categorical_attributes(thyroid_data)


    print("Attribute Data Types:\n", attribute_types)

    print("\nMissing Values in Each Attribute:\n", missing_values)

    print("\nMinimum Values of Numeric Attributes:\n", min_values)
    print("\nMaximum Values of Numeric Attributes:\n", max_values)

    print("\nMean of Numeric Attributes:\n", mean_values)
    print("\nVariance of Numeric Attributes:\n", variance_values)

    print("\nOutlier Count (IQR Method):")
    for key, value in outlier_info.items():
        print(f"{key}: {value}")

    print("\nCategorical Attributes Identified:")
    for col in categorical_attributes:
        print(col)

    print("\nEncoding Recommendation:")
    print("• Nominal variables → One-Hot Encoding")
    print("• Ordinal variables → Label Encoding")



if __name__ == "__main__":
    main()
