import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def statistical_analysis(df_train, df_test, validation_size=0.2):
    # Seleziona le colonne delle feature, escludendo 'id' e 'sii'
    feature_columns = df_train.columns.difference(['id', 'sii'])
    
    # Suddividi il training set in un vero training set e un validation set
    train_data, val_data = train_test_split(df_train, test_size=validation_size, random_state=42)

    # Standardizza i dati di training
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data[feature_columns])
    
    # Applica la PCA sui dati di training
    pca = PCA()
    pca_train_data = pca.fit_transform(scaled_train_data)
    
    # Calcola la varianza cumulativa per determinare il numero di componenti che spiegano il 90% della varianza
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    num_components_to_keep = np.argmax(cumulative_variance >= 0.9) + 1
    
    # Filtra le componenti principali che spiegano almeno il 90% della varianza nel training set
    pca_train_selected = pca_train_data[:, :num_components_to_keep]

    # Trasforma il validation set e il test set con lo stesso standardizzatore e PCA addestrati sul training set
    scaled_val_data = scaler.transform(val_data[feature_columns])
    scaled_test_data = scaler.transform(df_test[feature_columns])
    
    pca_val_data = pca.transform(scaled_val_data)
    pca_test_data = pca.transform(scaled_test_data)
    
    pca_val_selected = pca_val_data[:, :num_components_to_keep]
    pca_test_selected = pca_test_data[:, :num_components_to_keep]

    # Converti le componenti principali selezionate in DataFrame per training, validation e test, mantenendo gli indici originali
    df_pca_train_selected = pd.DataFrame(pca_train_selected, columns=[f'PC{i+1}' for i in range(num_components_to_keep)], index=train_data.index)
    df_pca_val_selected = pd.DataFrame(pca_val_selected, columns=[f'PC{i+1}' for i in range(num_components_to_keep)], index=val_data.index)
    df_pca_test_selected = pd.DataFrame(pca_test_selected, columns=[f'PC{i+1}' for i in range(num_components_to_keep)], index=df_test.index)
    
    return df_pca_train_selected, df_pca_val_selected, df_pca_test_selected
