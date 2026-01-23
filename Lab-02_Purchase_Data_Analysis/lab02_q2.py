# Lab Session 02 - A2
# Customer Classification using NumPy and Scikit-learn

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#  FUNCTION DEFINITIONS

def load_purchase_data(file_path):
    """
    Loads the Purchase data worksheet from the Excel file
    """
    return pd.read_excel(file_path, sheet_name="Purchase data")


def prepare_features_and_labels(dataframe):
    """
    Prepares feature matrix X and class labels y
    RICH -> 1, POOR -> 0
    """
    feature_matrix = dataframe[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    
    # Create class labels based on payment condition
    class_labels = np.where(dataframe["Payment (Rs)"] > 200, 1, 0)
    
    return feature_matrix, class_labels


def train_classifier(feature_matrix, class_labels):
    """
    Trains a logistic regression classifier
    """
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, class_labels, test_size=0.3, random_state=42
    )
    
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    predictions = classifier.predict(X_test)
    
    return classifier, y_test, predictions


# MAIN FUNCTION 

def main():
    file_path = "Purchase Data.xlsx"
    
    # Load data
    purchase_data = load_purchase_data(file_path)
    
    # Prepare X and y
    X, y = prepare_features_and_labels(purchase_data)
    
    # Train classifier and predict
    model, y_test, y_pred = train_classifier(X, y)
    
    # PRINT STATEMENTS (ONLY HERE) 
    
    print("Customer Classification Results")
    print("Accuracy of the model:", accuracy_score(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["POOR", "RICH"]))


#EXECUTION 

if __name__ == "__main__":
    main()
