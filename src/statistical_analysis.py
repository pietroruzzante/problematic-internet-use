import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter

def statistical_analysis(df_train, df_test, validation_size=0.2, num_components=10, smote_strategy="auto"):
    """
    Esegue la riduzione dimensionale tramite PCA e il bilanciamento del training set utilizzando SMOTE per gestire
    dataset sbilanciati.

    Parametri:
    - df_train (pd.DataFrame): Dataset di training con colonne 'id', 'sii' e feature.
    - df_test (pd.DataFrame): Dataset di test con colonne 'id' e feature.
    - validation_size (float): Proporzione di dati di training da utilizzare per la validation (default: 0.2).
    - num_components (int): Numero di componenti principali da mantenere tramite PCA (default: 10).
    - smote_strategy (str o dict): Strategia per bilanciare le classi con SMOTE (default: "auto").

    Ritorna:
    - df_train_pca (pd.DataFrame): Training set ridotto con le componenti principali.
    - y_train_balanced (pd.Series): Target del training set bilanciato.
    - df_val_pca (pd.DataFrame): Validation set ridotto con le componenti principali.
    - y_val (pd.Series): Target del validation set.
    - df_test_pca (pd.DataFrame): Test set ridotto con le componenti principali.
    """

    # Seleziona le colonne delle feature, escludendo 'id' e 'sii'
    feature_columns = df_train.columns.difference(['id', 'sii'])
    target_column = 'sii'

    # Suddividi il training set in un vero training set e un validation set
    train_data, val_data = train_test_split(
        df_train, 
        test_size=validation_size, 
        random_state=42, 
        stratify=df_train[target_column]
    )
    
    # Estrai le feature e i target
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_val = val_data[feature_columns]
    y_val = val_data[target_column]
    X_test = df_test[feature_columns]

    # Standardizza i dati
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Applica SMOTE per bilanciare il training set
    print(f"Distribuzione originale delle classi nel training set: {Counter(y_train)}")
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"Distribuzione dopo SMOTE: {Counter(y_train_balanced)}")

    # Applica la PCA per ridurre le dimensioni
    print(f"Applicazione PCA per ridurre le dimensioni a {num_components} componenti principali...")
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train_balanced)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Recupera la varianza spiegata
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Varianza spiegata totale dalle prime {num_components} componenti: {explained_variance:.2f}%")

    # Converti i dati PCA in DataFrame
    pca_columns = [f"PC{i+1}" for i in range(num_components)]
    df_train_pca = pd.DataFrame(X_train_pca, columns=pca_columns)
    df_val_pca = pd.DataFrame(X_val_pca, columns=pca_columns, index=val_data.index)
    df_test_pca = pd.DataFrame(X_test_pca, columns=pca_columns, index=df_test.index)

    # Ritorna i DataFrame con le componenti principali e i target
    return df_train_pca, y_train_balanced, df_val_pca, y_val, df_test_pca
