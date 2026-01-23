# A2: Intraclass Spread and Interclass Distance
# Dataset: WineQT.csv


import numpy as np
import pandas as pd


def compute_mean(vector):
    """
    Computes mean of a 1D vector
    """
    return sum(vector) / len(vector)


def compute_variance(vector):
    """
    Computes variance of a 1D vector
    """
    mean_value = compute_mean(vector)
    squared_diff_sum = 0.0
    for value in vector:
        squared_diff_sum += (value - mean_value) ** 2
    return squared_diff_sum / len(vector)


def compute_standard_deviation(vector):
    """
    Computes standard deviation of a 1D vector
    """
    return compute_variance(vector) ** 0.5


def compute_mean_matrix(data_matrix):
    """
    Computes mean vector for a matrix (column-wise)
    """
    mean_vector = []
    for col in range(data_matrix.shape[1]):
        mean_vector.append(compute_mean(data_matrix[:, col]))
    return np.array(mean_vector)


def compute_std_matrix(data_matrix):
    """
    Computes standard deviation vector for a matrix (column-wise)
    """
    std_vector = []
    for col in range(data_matrix.shape[1]):
        std_vector.append(compute_standard_deviation(data_matrix[:, col]))
    return np.array(std_vector)


def compute_interclass_distance(centroid_1, centroid_2):
    """
    Computes Euclidean distance between two centroids
    """
    return np.linalg.norm(centroid_1 - centroid_2)


# MAIN PROGRAM 

# Load dataset (use raw string to avoid unicode error)
data = pd.read_csv(
    r"WineQT.csv"
)

# Drop ID column if present
data = data.drop(columns=["Id"], errors="ignore")

# Choose two classes (quality = 5 and quality = 6)
class_1 = data[data["quality"] == 5].drop(columns=["quality"]).values
class_2 = data[data["quality"] == 6].drop(columns=["quality"]).values

# Compute Centroids (Means) 
centroid_class_1 = compute_mean_matrix(class_1)
centroid_class_2 = compute_mean_matrix(class_2)

#  Compute Intraclass Spread 
spread_class_1 = compute_std_matrix(class_1)
spread_class_2 = compute_std_matrix(class_2)

#  Compute Interclass Distance
interclass_distance = compute_interclass_distance(
    centroid_class_1, centroid_class_2
)


print("CLASS 1 (Quality = 5)")
print("Centroid (Mean Vector):", centroid_class_1)
print("Intraclass Spread (Std Dev Vector):", spread_class_1)

print("\nCLASS 2 (Quality = 6)")
print("Centroid (Mean Vector):", centroid_class_2)
print("Intraclass Spread (Std Dev Vector):", spread_class_2)

print("\nINTERCLASS DISTANCE")
print("Euclidean Distance Between Class Centroids:", interclass_distance)
