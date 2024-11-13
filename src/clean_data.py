from sklearn.impute import SimpleImputer

def clean_data(df_train_filtered, df_test_filtered):
    # Estrazione colonna target e id per il training set
    sii_column_train = df_train_filtered['sii']
    id_column_train = df_train_filtered['id']
    
    # Pulizia del dataset di training
    # Rimozione righe con troppi NaN nel dataset di training
    rows_to_keep_train = df_train_filtered.isna().sum(axis=1) <= 6
    df_train_cleaned = df_train_filtered.loc[rows_to_keep_train].copy()

    # Colonne numeriche nel dataset di training
    numerical_columns_train = df_train_cleaned.select_dtypes(include=['float64', 'int64']).columns

    # Data imputation per il dataset di training
    num_imputer_train = SimpleImputer(strategy='median')
    df_train_cleaned[numerical_columns_train] = num_imputer_train.fit_transform(df_train_cleaned[numerical_columns_train])

    # Ritorno al df_train completo
    df_train_cleaned['sii'] = sii_column_train.loc[rows_to_keep_train].values
    df_train_cleaned['id'] = id_column_train.loc[rows_to_keep_train].values

    # Pulizia del dataset di test (senza eliminazione di righe)
    id_column_test = df_test_filtered['id']  # Estrazione colonna id per il test set

    # Creazione di una copia del dataset di test per evitare modifiche dirette
    df_test_cleaned = df_test_filtered.copy()

    # Colonne numeriche nel dataset di test
    numerical_columns_test = df_test_cleaned.select_dtypes(include=['float64', 'int64']).columns

    # Data imputation per il dataset di test (senza eliminazione di righe)
    num_imputer_test = SimpleImputer(strategy='median')
    df_test_cleaned[numerical_columns_test] = num_imputer_test.fit_transform(df_test_cleaned[numerical_columns_test])

    # Ritorno al df_test completo
    df_test_cleaned['id'] = id_column_test.values

    return df_train_cleaned, df_test_cleaned
