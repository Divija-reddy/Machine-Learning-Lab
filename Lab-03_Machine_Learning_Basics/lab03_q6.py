# A6: Train-Test Split
# Dataset: WineQT.csv
# Classes Chosen: quality = 5 and quality = 6

import pandas as pd
from sklearn.model_selection import train_test_split

#  FUNCTION DEFINITIONS 

def split_dataset(features, labels, test_ratio):
    """
    Splits the dataset into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_ratio,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


#  MAIN PROGRAM 

# Load dataset (raw string avoids unicode error)
data = pd.read_csv(
    r"WineQT.csv"
)

# Drop ID column if present
data = data.drop(columns=["Id"], errors="ignore")

# Select Only Two Classes 
binary_data = data[data["quality"].isin([5, 6])]

# Separate features (X) and labels (y)
X = binary_data.drop(columns=["quality"])
y = binary_data["quality"]

# Train-Test Split 

X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.3)


print("TRAIN-TEST SPLIT DETAILS")
print("Total samples:", len(binary_data))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nFeature matrix shape (Train):", X_train.shape)
print("Feature matrix shape (Test):", X_test.shape)

print("\nClass labels in training set:")
print(y_train.value_counts())

print("\nClass labels in testing set:")
print(y_test.value_counts())
