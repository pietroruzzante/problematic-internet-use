from sklearn.impute import SimpleImputer

def clean_data(df_filtered, col_name):

    # Estrazione colonna target e id
    sii_column = df_filtered['sii']
    id_column = df_filtered['id']
    
    #Rimozione righe con troppi NaN
    rows_to_keep = df_filtered.isna().sum(axis=1) <= 4
    df_filtered_cleaned = df_filtered.loc[rows_to_keep].copy()  

    # Colonne numeriche
    numerical_columns = df_filtered_cleaned.select_dtypes(include=['float64', 'int64']).columns
    #print("Colonne numeriche:", numerical_columns)

    # Data imputation
    num_imputer = SimpleImputer(strategy='median')
    df_filtered_cleaned[numerical_columns] = num_imputer.fit_transform(df_filtered_cleaned[numerical_columns])


    # Ritorno al df completo
    df_filtered_cleaned['sii'] = sii_column.loc[rows_to_keep].values
    df_filtered_cleaned['id'] = id_column.loc[rows_to_keep].values

    """ DEBUG
    # Mostra le prime righe del DataFrame pulito e verifica la presenza di NaN solo in sii
    print(df_filtered_cleaned.head())
    print(df_filtered_cleaned.shape)
    print("Valori NaN dopo l'imputazione:\n", df_filtered_cleaned.isna().sum())
    """
    return df_filtered_cleaned