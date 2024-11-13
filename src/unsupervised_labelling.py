import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def unsupervised_labelling(df):
    
    sii_unique_values = df['sii'].dropna().unique()
    n_clusters = len(sii_unique_values)
    #print("Numero di cluster stimato (basato sui valori unici di 'sii'):", n_clusters)

    # Separare le righe con e senza 'sii'
    df_with_sii = df.dropna(subset=['sii']).copy()
    df_without_sii = df[df['sii'].isna()].copy()

    # Rimuove 'id' e 'sii' per il clustering
    feature_columns = df.columns.difference(['id', 'sii'])

    # Standardizzazione dei dati
    scaler = StandardScaler()
    df_with_sii_scaled = scaler.fit_transform(df_with_sii[feature_columns])
    df_without_sii_scaled = scaler.transform(df_without_sii[feature_columns])

    # Applica il clustering K-means sui dati senza 'sii'
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_with_sii_scaled)  # Modello addestrato sui dati con etichette
    
    # Predice i cluster per i dati senza 'sii'
    clusters_without_sii = kmeans.predict(df_without_sii_scaled)

    # Assegna le etichette del cluster come nuove etichette per 'sii'
    df_without_sii['sii'] = clusters_without_sii

    # Combina il dataset etichettato con quello precedente
    df_combined = pd.concat([df_with_sii, df_without_sii], axis=0).sort_index()

    """ DEBUG
    # verifica l'assenza di NaN
    print("Il Dataset non contiene valori NaN." if df_combined.isna().sum().sum() == 0 else "Il Dataset contiene valori NaN.")
    """
    return df_combined