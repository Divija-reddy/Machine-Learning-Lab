import pandas as pd
import matplotlib.pyplot as plt
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


# Function to train and evaluate kNN
def knn_accuracy(X_train, y_train, X_test, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy


# Function to compute accuracy for k = 1 to 11
def accuracy_for_k_values(X_train, y_train, X_test, y_test):
    k_values = list(range(1, 12))
    accuracies = []

    for k in k_values:
        acc = knn_accuracy(X_train, y_train, X_test, y_test, k)
        accuracies.append(acc)

    return k_values, accuracies


if __name__ == "__main__":

    file_path = "WineQT.csv"

    # Load and split data
    X, y = load_dataset(file_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # NN (k = 1)
    accuracy_k1 = knn_accuracy(X_train, y_train, X_test, y_test, k=1)

    # kNN (k = 3)
    accuracy_k3 = knn_accuracy(X_train, y_train, X_test, y_test, k=3)

    # Accuracy for k = 1 to 11
    k_values, accuracies = accuracy_for_k_values(
        X_train, y_train, X_test, y_test
    )

    # Print comparison
    print("Accuracy for NN (k = 1):", accuracy_k1)
    print("Accuracy for kNN (k = 3):", accuracy_k3)

    # Plot accuracy vs k
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("kNN Accuracy for k = 1 to 11")
    plt.grid(True)
    plt.show()
