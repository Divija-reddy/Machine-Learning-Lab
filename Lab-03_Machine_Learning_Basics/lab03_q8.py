import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Function to load dataset and separate features and labels
def load_dataset(file_path):
    data = pd.read_csv(file_path)

    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])

    X = data.drop(columns=['quality']).values
    y = data['quality'].values

    return X, y


# Function to split dataset into train and test sets
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# Function to train k-NN classifier
def train_knn_classifier(X_train, y_train, k_value):
    neigh = KNeighborsClassifier(n_neighbors=k_value)
    neigh.fit(X_train, y_train)
    return neigh


# Function to test accuracy using test set
def test_accuracy(neigh, X_test, y_test):
    accuracy = neigh.score(X_test, y_test)
    return accuracy


if __name__ == "__main__":

    file_path = "WineQT.csv"

    X, y = load_dataset(file_path)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train k-NN with k = 3
    neigh = train_knn_classifier(X_train, y_train, k_value=3)

    # Test accuracy
    accuracy = test_accuracy(neigh, X_test, y_test)

    print("k-NN Test Accuracy:", accuracy)
