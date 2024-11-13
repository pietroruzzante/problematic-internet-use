import sys
import os
import pandas as pd
from preprocessing import preprocessing
from clean_data import clean_data
from unsupervised_labelling import unsupervised_labelling
from statistical_analysis import statistical_analysis
from model_training import train_and_evaluate_model, generate_test_predictions

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # Carica il dataset
    df_train = pd.read_csv("data/raw/train.csv")
    df_test = pd.read_csv("data/raw/test.csv")

    # Preprocessing dei dati
    df_train_filtered, df_test_filtered = preprocessing(df_train, df_test)

    # Pulizia dei dati
    df_train_filtered_cleaned,  df_test_filtered_cleaned = clean_data(df_train_filtered, df_test_filtered)

    # Unsupervised Labelling (KMeans)
    df_train_final = unsupervised_labelling(df_train_filtered_cleaned)

    # Analisi statistica e PCA con creazione del validation set
    df_train_PCA, df_val_PCA, df_test_PCA = statistical_analysis(df_train_final.drop(columns=['id']), df_test_filtered_cleaned.drop(columns=['id']), validation_size=0.2)

    # Estrazione dei target
    y_train = df_train_final.loc[df_train_PCA.index, 'sii']
    y_val = df_train_final.loc[df_val_PCA.index, 'sii']

    # Allenamento e valutazione del modello sul validation set
    model, _ = train_and_evaluate_model(df_train_PCA, y_train, df_val_PCA, y_val, model_type="random_forest")

    # Genera le previsioni finali sul test set (senza valutazione)
    generate_test_predictions(model, df_test_PCA, model_type="random_forest")


if __name__ == "__main__":
    main()
