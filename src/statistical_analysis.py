import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def statistical_analysis(df):
    # Seleziona le colonne delle feature, escludendo 'id' e 'sii'
    feature_columns = df.columns.difference(['id', 'sii'])
    
    # Standardizza i dati
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])

    # Applica la PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    
    # Calcola la varianza cumulativa per determinare il numero di componenti che spiegano il 90% della varianza
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    num_components_to_keep = np.argmax(cumulative_variance >= 0.9) + 1
    
    #print(f"Numero di componenti principali per spiegare il 90% della varianza: {num_components_to_keep}")
    
    # Filtra le componenti principali che spiegano almeno il 90% della varianza
    pca_selected = pca_data[:, :num_components_to_keep]

    # Converti le componenti principali selezionate in un DataFrame
    df_pca_selected = pd.DataFrame(pca_selected, columns=[f'PC{i+1}' for i in range(num_components_to_keep)])
    
    return df_pca_selected
