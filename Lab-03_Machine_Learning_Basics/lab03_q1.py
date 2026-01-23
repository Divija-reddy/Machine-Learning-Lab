# A1: Vector Operations using Project Dataset
# Dataset: WineQT.csv

import numpy as np
import pandas as pd


def compute_dot_product(vector_a, vector_b):
    """
    Computes dot product of two N-dimensional vectors
    """
    dot_product = 0.0
    for i in range(len(vector_a)):
        dot_product += vector_a[i] * vector_b[i]
    return dot_product


def compute_euclidean_norm(vector):
    """
    Computes Euclidean norm (L2 norm) of a vector
    """
    sum_of_squares = 0.0
    for value in vector:
        sum_of_squares += value ** 2
    return sum_of_squares ** 0.5


# Load the dataset
data = pd.read_csv("WineQT.csv")

# Drop non-feature columns if present
data = data.drop(columns=["Id"], errors="ignore")

# Select two sample vectors from the dataset
vector_A = data.iloc[0].values
vector_B = data.iloc[1].values


custom_dot = compute_dot_product(vector_A, vector_B)
custom_norm_A = compute_euclidean_norm(vector_A)
custom_norm_B = compute_euclidean_norm(vector_B)


numpy_dot = np.dot(vector_A, vector_B)
numpy_norm_A = np.linalg.norm(vector_A)
numpy_norm_B = np.linalg.norm(vector_B)


print("CUSTOM IMPLEMENTATION RESULTS")
print("Dot Product:", custom_dot)
print("Euclidean Norm of Vector A:", custom_norm_A)
print("Euclidean Norm of Vector B:", custom_norm_B)

print("\nNUMPY IMPLEMENTATION RESULTS")
print("Dot Product:", numpy_dot)
print("Euclidean Norm of Vector A:", numpy_norm_A)
print("Euclidean Norm of Vector B:", numpy_norm_B)

print("\nCOMPARISON")
print("Dot Product Match:", np.isclose(custom_dot, numpy_dot))
print("Norm A Match:", np.isclose(custom_norm_A, numpy_norm_A))
print("Norm B Match:", np.isclose(custom_norm_B, numpy_norm_B))
