# Lab Session 02 - A6
# Cosine Similarity Measure

import pandas as pd
import numpy as np

#  FUNCTION DEFINITIONS 

def load_thyroid_data(file_path):
    """
    Loads thyroid0387_UCI worksheet from Excel file
    """
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def extract_numeric_features(dataframe):
    """
    Extracts only numeric attributes from the dataset
    """
    numeric_dataframe = dataframe.select_dtypes(include=[np.number])
    return numeric_dataframe


def cosine_similarity(vector_a, vector_b):
    """
    Calculates cosine similarity between two vectors safely
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)



def main():
    file_path = "Purchase Data.xlsx"

    # Load dataset
    thyroid_data = load_thyroid_data(file_path)

    # Extract numeric attributes only
    numeric_data = extract_numeric_features(thyroid_data)

    # Take complete feature vectors of first two observations
    vector_1 = numeric_data.iloc[0].values
    vector_2 = numeric_data.iloc[1].values

    # Compute cosine similarity
    cosine_value = cosine_similarity(vector_1, vector_2)


    print("Number of numeric attributes used:", len(vector_1))

    print("\nCosine Similarity between first two observations:")
    print(cosine_value)

    print("\nInference:")
    print("Cosine similarity measures the angle between vectors,")
    print("and is insensitive to magnitude but sensitive to direction.")


if __name__ == "__main__":
    main()
