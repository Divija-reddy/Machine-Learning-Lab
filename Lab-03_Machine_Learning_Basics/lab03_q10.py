# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Function to load dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)

    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])

    X = data.drop(columns=['quality']).values
    y = data['quality'].values

    return X, y


# Function to split dataset
def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to train sklearn kNN
def train_sklearn_knn(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Manual kNN prediction function
def manual_knn_predict(X_train, y_train, X_test, k):
    predictions = []

    for test_point in X_test:
        distances = []

        for i in range(len(X_train)):
            distance = euclidean_distance(test_point, X_train[i])
            distances.append((distance, y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for (_, label) in distances[:k]]

        predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(predicted_label)

    return np.array(predictions)


# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# Main program
if __name__ == "__main__":

    file_path = "WineQT.csv"

    # Load and split data
    X, y = load_dataset(file_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # sklearn kNN
    sklearn_knn = train_sklearn_knn(X_train, y_train, k=3)
    sklearn_accuracy = sklearn_knn.score(X_test, y_test)

    # Manual kNN
    manual_predictions = manual_knn_predict(X_train, y_train, X_test, k=3)
    manual_accuracy = calculate_accuracy(y_test, manual_predictions)

    # Output comparison
    print("Sklearn kNN Accuracy:", sklearn_accuracy)
    print("Manual kNN Accuracy :", manual_accuracy)
