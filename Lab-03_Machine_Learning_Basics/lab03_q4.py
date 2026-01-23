# A4: Minkowski Distance for p = 1 to 10
# Dataset: WineQT.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  FUNCTION DEFINITIONS

def compute_minkowski_distance(vector_a, vector_b, p):
    """
    Computes Minkowski distance between two vectors for a given p
    """
    distance_sum = 0.0
    for i in range(len(vector_a)):
        distance_sum += abs(vector_a[i] - vector_b[i]) ** p
    return distance_sum ** (1 / p)


def compute_distances_for_p_range(vector_a, vector_b, p_values):
    """
    Computes Minkowski distances for multiple p values
    """
    distances = []
    for p in p_values:
        distances.append(compute_minkowski_distance(vector_a, vector_b, p))
    return distances


def plot_distance_vs_p(p_values, distances):
    """
    Plots Minkowski distance vs p
    """
    plt.plot(p_values, distances, marker='o')
    plt.xlabel("p value")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs p")
    plt.grid(True)
    plt.show()


#  MAIN PROGRAM 
# Load dataset (raw string avoids unicode error)
data = pd.read_csv(
    r"WineQT.csv"
)

# Drop ID column if present
data = data.drop(columns=["Id"], errors="ignore")

# Remove class label
features = data.drop(columns=["quality"])

# Select any two feature vectors
vector_1 = features.iloc[0].values
vector_2 = features.iloc[1].values

# p values from 1 to 10
p_values = list(range(1, 11))

# Compute Minkowski distances
minkowski_distances = compute_distances_for_p_range(
    vector_1, vector_2, p_values
)

# Plot the results
plot_distance_vs_p(p_values, minkowski_distances)


print("Minkowski Distance Values")
for p, d in zip(p_values, minkowski_distances):
    print(f"p = {p} -> Distance = {d}")
