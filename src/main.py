import pandas as pd
import sys 
from src.preprocessing import preprocessing
from src.clean_data import clean_data
from src.unsupervised_labelling import unsupervised_labelling
from src.statistical_analysis import statistical_analysis

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