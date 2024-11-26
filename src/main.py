import sys
import os
import pandas as pd
from preprocessing import preprocessing
from clean_data import clean_data
from unsupervised_labelling import unsupervised_labelling
from statistical_analysis import statistical_analysis
from model_training import train_and_evaluate_models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():

    active_preprocessing = True

    # Definisci i percorsi per i file processati
    processed_dir = "data/processed"
    train_selected_path = os.path.join(processed_dir, "train_selected.csv")
    val_selected_path = os.path.join(processed_dir, "val_selected.csv")
    test_selected_path = os.path.join(processed_dir, "test_selected.csv")
    train_target_path = os.path.join(processed_dir, "train_target.csv")
    val_target_path = os.path.join(processed_dir, "val_target.csv")

    # Controlla se i file processati esistono già
    if (all(os.path.exists(path) for path in [train_selected_path, val_selected_path, test_selected_path, train_target_path, val_target_path]))|active_preprocessing == False:
        print("File processati trovati. Salto il preprocessing e la pulizia.")
        # Carica i file processati
        df_train_selected = pd.read_csv(train_selected_path)
        df_val_selected = pd.read_csv(val_selected_path)
        df_test_selected = pd.read_csv(test_selected_path)
        y_train = pd.read_csv(train_target_path)['sii']
        y_val = pd.read_csv(val_target_path)['sii']
    else:
        print("File processati non trovati. Eseguo il preprocessing e la pulizia.")

        # Carica il dataset
        df_train = pd.read_csv("data/raw/train.csv")
        df_test = pd.read_csv("data/raw/test.csv")

        id_test = df_test['id']

        # Preprocessing dei dati
        df_train_filtered, df_test_filtered = preprocessing(df_train, df_test)

        # Pulizia dei dati
        df_train_filtered_cleaned, df_test_filtered_cleaned = clean_data(df_train_filtered, df_test_filtered)

        # Salva i dataset puliti nella directory data/processed
        os.makedirs(processed_dir, exist_ok=True)
        df_train_filtered_cleaned.to_csv(os.path.join(processed_dir, "cleaned_train.csv"), index=False)
        df_test_filtered_cleaned.to_csv(os.path.join(processed_dir, "cleaned_test.csv"), index=False)

        # Unsupervised Labelling (KMeans)
        df_train_final = unsupervised_labelling(df_train_filtered_cleaned)

        # Salva il dataset con etichette generate
        df_train_final.to_csv(os.path.join(processed_dir, "labeled_train.csv"), index=False)

        # Selezione delle feature e suddivisione dei dataset
        df_train_selected, y_train, df_val_selected, y_val, df_test_selected = statistical_analysis(
            df_train_final, df_test_filtered_cleaned, validation_size=0.20, num_components=13, smote_strategy="auto"
        )

        # Salva i dataset selezionati
        df_train_selected.to_csv(train_selected_path, index=False)
        df_val_selected.to_csv(val_selected_path, index=False)
        df_test_selected.to_csv(test_selected_path, index=False)

        # Salva i target del training e validation set
        pd.DataFrame({'sii': y_train}).to_csv(train_target_path, index=False)
        pd.DataFrame({'sii': y_val}).to_csv(val_target_path, index=False)

    ## Verifica le dimensioni dei dataset
    print(f"Training set bilanciato: {df_train_selected.shape}, Target bilanciati: {pd.Series(y_train).value_counts()}")
    print(f"Validation set: {df_val_selected.shape}, Target: {y_val.value_counts()}")
    print(f"Test set: {df_test_selected.shape}")

    # Allenamento e valutazione del modello sul validation set

    result, best_model = train_and_evaluate_models(df_train_selected, y_train, df_val_selected, y_val)

    test_results = best_model.predict(df_test_selected)

    results_df = pd.DataFrame({
    'id': id_test.values,  # Prende i valori degli ID
    'sii': test_results.astype(int)  # Associa i risultati predetti
    })

    # Mostra i primi risultati per verifica
    print(results_df.head())

    # Salva il file CSV finale (se richiesto)
    os.makedirs(processed_dir, exist_ok=True)
    results_df.to_csv(os.path.join( "submission.csv"), index=False)

if __name__ == "__main__":
    main()
