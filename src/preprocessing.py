def preprocessing(df):
    columns_to_keep = [
        'id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'BIA-BIA_Activity_Level_num',
        'CGAS-CGAS_Score', 'PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total', 
        'PCIAT-PCIAT_Total', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 
        'PreInt_EduHx-computerinternet_hoursday', 'sii'
    ]
    df_filtered = df[columns_to_keep].copy()  # Crea una copia esplicita del DataFrame selezionato

    # 'PAQ_Total' combina 'PAQ_A-PAQ_A_Total' e 'PAQ_C-PAQ_C_Total'
    df_filtered.loc[:, 'PAQ_Total'] = df_filtered['PAQ_A-PAQ_A_Total'].combine_first(df_filtered['PAQ_C-PAQ_C_Total'])
    
    # Rimuovi le colonne originali
    df_filtered = df_filtered.drop(columns=['PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total'])
    col_name = df_filtered.columns.tolist()

    return df_filtered, col_name
