import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

def train_and_evaluate_model(X_train, y_train, X_val, y_val, model_type="random_forest"):
    # Scegli il modello basato sul parametro model_type
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "svm":
        model = SVC(random_state=42)
    else:
        raise ValueError("Model type non riconosciuto. Usa 'random_forest', 'logistic_regression' o 'svm'.")

    # Addestra il modello
    model.fit(X_train, y_train)

    # Valutazione sul dataset di validation
    y_pred_val = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    report = classification_report(y_val, y_pred_val)

    # Stampa le metriche di valutazione
    print(f"\nAccuracy del modello {model_type} sul validation set: {accuracy:.2f}")
    print(f"Report di classificazione:\n{report}")

    # Salva il modello addestrato
    model_filename = f"models/{model_type}_model.joblib"
    dump(model, model_filename)
    print(f"Modello salvato come {model_filename}")

    return model, accuracy

def generate_test_predictions(model, X_test, model_type="random_forest"):
    # Genera le previsioni sul dataset di test finale (senza target)
    y_test_pred = model.predict(X_test)
    return y_test_pred
