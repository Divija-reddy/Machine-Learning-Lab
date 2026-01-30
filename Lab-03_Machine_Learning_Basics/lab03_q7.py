import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# Function to load dataset and separate features and labels
def load_dataset(file_path):
    data = pd.read_csv(file_path)

    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])

    X = data.drop(columns=['quality']).values
    y = data['quality'].values

    return X, y


# Function to train k-NN classifier (k = 3)
def train_knn_classifier(X, y, k_value):
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X, y)
    return knn_model

if __name__ == "__main__":

    file_path = "WineQT.csv"   # dataset in same folder

    # Load dataset
    X, y = load_dataset(file_path)

    # Train k-NN classifier with k = 3
    knn_model = train_knn_classifier(X, y, k_value=3)

    print("k-NN classifier trained successfully with k = 3")
