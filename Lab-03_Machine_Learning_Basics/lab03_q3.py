
# A3: Density Pattern, Histogram, Mean & Variance
# Dataset: WineQT.csv
# Feature Chosen: alcohol

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def compute_histogram(vector, number_of_bins):
    """
    Computes histogram data using buckets
    """
    histogram_values, bin_edges = np.histogram(vector, bins=number_of_bins)
    return histogram_values, bin_edges


def plot_histogram(vector, number_of_bins, feature_name):
    """
    Plots histogram for the given feature
    """
    plt.hist(vector, bins=number_of_bins)
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {feature_name}")
    plt.grid(True)
    plt.show()


#MAIN PROGRAM 

# Load dataset (raw string avoids unicode error)
data = pd.read_csv(
    r"WineQT.csv"
)

# Drop ID column if present
data = data.drop(columns=["Id"], errors="ignore")

# Select one feature (Alcohol)
feature_data = data["alcohol"].values

# Number of histogram buckets
bins = 10

# Histogram Data

hist_values, bin_edges = compute_histogram(feature_data, bins)

# Mean & Variance 

mean_value = compute_mean(feature_data)
variance_value = compute_variance(feature_data)

# Plot Histogram 

plot_histogram(feature_data, bins, "Alcohol")


print("FEATURE ANALYSIS: Alcohol")
print("Mean:", mean_value)
print("Variance:", variance_value)
print("\nHistogram Values:", hist_values)
print("Bin Edges:", bin_edges)
