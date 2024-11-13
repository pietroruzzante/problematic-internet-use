import sys
import os
import pandas as pd
from preprocessing import preprocessing
from clean_data import clean_data
from unsupervised_labelling import unsupervised_labelling
from statistical_analysis import statistical_analysis

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # Carica il dataset
    df = pd.read_csv("data/raw/train.csv")

    # Preprocessing dei dati
    df_filtered, col_name = preprocessing(df)

    # Pulizia dei dati
    df_filtered_cleaned = clean_data(df_filtered, col_name)

    # Unsupervised Labelling (KMeans)
    df_final = unsupervised_labelling(df_filtered_cleaned)

    # Analisi statistica
    df_PCA = statistical_analysis(df_final.drop(columns=['id']))

    return 0 


if __name__ == "__main__":
    main()  