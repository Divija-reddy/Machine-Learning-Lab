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


# Function to train kNN classifier
def train_knn_classifier(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


# Function to train matrix inversion classifier (least squares)
def train_matrix_inversion_classifier(X_train, y_train):
    X_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y_train
    return weights


# Function to predict using matrix inversion classifier
def predict_matrix_inversion(X_test, weights):
    X_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    predictions = X_bias @ weights
    return np.rint(predictions).astype(int)


# Function to compute accuracy
def compute_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == "__main__":

    file_path = "WineQT.csv"

    # Load and split data
    X, y = load_dataset(file_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # kNN classifier (k = 3)
    knn_model = train_knn_classifier(X_train, y_train, k=3)
    knn_predictions = knn_model.predict(X_test)
    knn_accuracy = compute_accuracy(y_test, knn_predictions)

    # Matrix inversion classifier
    weights = train_matrix_inversion_classifier(X_train, y_train)
    mi_predictions = predict_matrix_inversion(X_test, weights)
    mi_accuracy = compute_accuracy(y_test, mi_predictions)

    print("kNN Classifier Accuracy           :", knn_accuracy)
    print("Matrix Inversion Classifier Accuracy:", mi_accuracy)
