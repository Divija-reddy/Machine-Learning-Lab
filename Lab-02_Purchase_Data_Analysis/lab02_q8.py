# Lab Session 02 - A8
# Data Imputation using Central Tendencies

import pandas as pd
import numpy as np

#  FUNCTION DEFINITIONS

def load_thyroid_data(file_path):
    """
    Loads thyroid0387_UCI worksheet from Excel file
    """
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def detect_outliers_iqr(series):
    """
    Detects whether a numeric column contains outliers using IQR
    Returns True if outliers exist, else False
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return any((series < lower_bound) | (series > upper_bound))


def impute_numeric_column(series):
    """
    Imputes numeric column using mean or median
    """
    if detect_outliers_iqr(series.dropna()):
        return series.fillna(series.median())
    else:
        return series.fillna(series.mean())


def impute_categorical_column(series):
    """
    Imputes categorical column using mode
    """
    mode_value = series.mode()
    if not mode_value.empty:
        return series.fillna(mode_value[0])
    return series


def impute_missing_values(dataframe):
    """
    Imputes missing values in the dataframe
    """
    imputed_dataframe = dataframe.copy()

    for column in imputed_dataframe.columns:
        if imputed_dataframe[column].dtype in [np.float64, np.int64]:
            imputed_dataframe[column] = impute_numeric_column(imputed_dataframe[column])
        else:
            imputed_dataframe[column] = impute_categorical_column(imputed_dataframe[column])

    return imputed_dataframe



def main():
    file_path = "Purchase Data.xlsx"

    # Load dataset
    thyroid_data = load_thyroid_data(file_path)

    # Count missing values before imputation
    missing_before = thyroid_data.isnull().sum()

    # Perform imputation
    imputed_data = impute_missing_values(thyroid_data)

    # Count missing values after imputation
    missing_after = imputed_data.isnull().sum()


    print("Missing Values Before Imputation:\n")
    print(missing_before)

    print("\nMissing Values After Imputation:\n")
    print(missing_after)



if __name__ == "__main__":
    main()
