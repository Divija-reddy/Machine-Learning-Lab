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


# Function to compute confusion matrix
def confusion_matrix_manual(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    size = len(labels)
    label_index = {label: index for index, label in enumerate(labels)}

    matrix = np.zeros((size, size), dtype=int)

    for true, pred in zip(y_true, y_pred):
        i = label_index[true]
        j = label_index[pred]
        matrix[i][j] += 1

    return matrix


# Function to compute accuracy
def accuracy_manual(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


# Function to compute precision (macro-averaged)
def precision_manual(conf_matrix):
    precisions = []
    for i in range(len(conf_matrix)):
        tp = conf_matrix[i][i]
        fp = np.sum(conf_matrix[:, i]) - tp
        if tp + fp == 0:
            precisions.append(0)
        else:
            precisions.append(tp / (tp + fp))
    return np.mean(precisions)


# Function to compute recall (macro-averaged)
def recall_manual(conf_matrix):
    recalls = []
    for i in range(len(conf_matrix)):
        tp = conf_matrix[i][i]
        fn = np.sum(conf_matrix[i, :]) - tp
        if tp + fn == 0:
            recalls.append(0)
        else:
            recalls.append(tp / (tp + fn))
    return np.mean(recalls)


# Function to compute F-beta score
def fbeta_manual(precision, recall, beta):
    if precision == 0 and recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


if __name__ == "__main__":

    file_path = "WineQT.csv"

    # Load and split data
    X, y = load_dataset(file_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train kNN classifier (k = 3)
    knn_model = train_knn_classifier(X_train, y_train, k=3)

    # Predict test labels
    y_pred = knn_model.predict(X_test)

    # Compute metrics
    conf_matrix = confusion_matrix_manual(y_test, y_pred)
    accuracy = accuracy_manual(y_test, y_pred)
    precision = precision_manual(conf_matrix)
    recall = recall_manual(conf_matrix)
    fbeta = fbeta_manual(precision, recall, beta=1)

    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", fbeta)
