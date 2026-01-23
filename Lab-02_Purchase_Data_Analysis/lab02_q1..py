# Lab Session 02 - A1
# Linear Algebra using NumPy
# Purchase Data Analysis

import pandas as pd
import numpy as np

# FUNCTION DEFINITIONS 

def load_purchase_data(file_path):
    """
    Loads the Purchase data worksheet from the Excel file
    """
    return pd.read_excel(file_path, sheet_name="Purchase data")


def segregate_features_and_output(dataframe):
    """
    Segregates feature matrix X and output vector y
    """
    feature_matrix = dataframe[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    output_vector = dataframe["Payment (Rs)"].values.reshape(-1, 1)
    return feature_matrix, output_vector


def calculate_matrix_rank(matrix):
    """
    Calculates rank of a matrix using NumPy
    """
    return np.linalg.matrix_rank(matrix)


def calculate_cost_per_product(feature_matrix, output_vector):
    """
    Calculates cost of each product using Moore-Penrose Pseudo-Inverse
    Xc = y  →  c = pinv(X)y
    """
    pseudo_inverse = np.linalg.pinv(feature_matrix)
    cost_vector = pseudo_inverse @ output_vector
    return cost_vector


#  MAIN FUNCTION 

def main():
    # Load data (Excel file in same folder as code)
    file_path = r"C:\Users\Divija\OneDrive\Documents\SEM-4\ML\LAB2\Purchase Data.xlsx"
    purchase_data = load_purchase_data(file_path)

    # Segregate X and y
    X, y = segregate_features_and_output(purchase_data)

    # Thinking outputs
    dimensionality = X.shape[1]
    number_of_vectors = X.shape[0]

    # Rank calculation
    rank_X = calculate_matrix_rank(X)

    # Cost calculation
    cost_per_product = calculate_cost_per_product(X, y)

    #  PRINT STATEMENTS (ONLY HERE) -

    print("Dimensionality of the vector space:", dimensionality)
    print("Number of vectors in the vector space:", number_of_vectors)
    print("Rank of the feature matrix:", rank_X)

    print("\nCost of each product:")
    print("Candies (Rs per unit):", cost_per_product[0][0])
    print("Mangoes (Rs per Kg):", cost_per_product[1][0])
    print("Milk Packets (Rs per packet):", cost_per_product[2][0])


if __name__ == "__main__":
    main()
