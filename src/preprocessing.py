from sklearn.impute import SimpleImputer

def preprocessing(df_train, df_test):
    # Definisci le colonne da mantenere, incluso 'sii' per il training set
    columns_to_keep = [
        'id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'BIA-BIA_Activity_Level_num',
        'CGAS-CGAS_Score', 'PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total', 
        'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 
        'PreInt_EduHx-computerinternet_hoursday', 'sii'
    ]

    # Filtra le colonne per il dataset di training
    df_train_filtered = df_train[columns_to_keep].copy()

    # Crea una nuova lista di colonne per il dataset di test, escludendo 'sii'
    columns_to_keep_test = [col for col in columns_to_keep if col != 'sii']
    df_test_filtered = df_test[columns_to_keep_test].copy()

    # Combina 'PAQ_A-PAQ_A_Total' e 'PAQ_C-PAQ_C_Total' in una nuova colonna 'PAQ_Total'
    df_train_filtered['PAQ_Total'] = df_train_filtered['PAQ_A-PAQ_A_Total'].combine_first(df_train_filtered['PAQ_C-PAQ_C_Total'])
    df_test_filtered['PAQ_Total'] = df_test_filtered['PAQ_A-PAQ_A_Total'].combine_first(df_test_filtered['PAQ_C-PAQ_C_Total'])

    # Rimuovi le colonne originali usate per creare 'PAQ_Total'
    df_train_filtered = df_train_filtered.drop(columns=['PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total'])
    df_test_filtered = df_test_filtered.drop(columns=['PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total'])

    # Raccogli i nomi delle colonne del dataset di training per usarli in altri processi
    col_name = df_train_filtered.columns.tolist()

    return df_train_filtered, df_test_filtered
