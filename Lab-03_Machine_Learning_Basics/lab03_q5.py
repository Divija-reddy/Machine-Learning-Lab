# A5: Comparison of Minkowski Distance
# Custom Implementation vs SciPy
# Dataset: WineQT.csv

import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski


def compute_minkowski_distance(vector_a, vector_b, p):
    """
    Computes Minkowski distance between two vectors for a given p
    """
    distance_sum = 0.0
    for i in range(len(vector_a)):
        distance_sum += abs(vector_a[i] - vector_b[i]) ** p
    return distance_sum ** (1 / p)


#MAIN PROGRAM 
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

# Choose p value
p_value = 3

# Distance Calculations
custom_distance = compute_minkowski_distance(vector_1, vector_2, p_value)
scipy_distance = minkowski(vector_1, vector_2, p_value)


print("MINKOWSKI DISTANCE COMPARISON")
print("p value:", p_value)

print("\nCustom Implementation Distance:", custom_distance)
print("SciPy Minkowski Distance:", scipy_distance)

print("\nDo both distances match?",
      np.isclose(custom_distance, scipy_distance))
