import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, cohen_kappa_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Quadratic Weighted Kappa function
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

"""
# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(16, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x
"""

def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    """
    Trains and evaluates Random Forest, Logistic Regression, SVM, and Neural Network models sequentially.
    """
    # StandardScaler for models requiring feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    results = {}
    best_model = None
    best_kappa = -np.inf  # Per tenere traccia del miglior modello in base a QWK

    scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)

    # Model 1: Random Forest
    print("\n--- Training Random Forest ---")
    param_grid_rf = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [None, 10, 20, 50],
        'criterion': ['gini', 'entropy'],
    }
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring=scorer)
    grid_search_rf.fit(X_train, y_train)

    best_rf = grid_search_rf.best_estimator_
    y_val_pred_rf = best_rf.predict(X_val)
    rf_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rf)
    results["Random Forest"] = {
        "Best Parameters": grid_search_rf.best_params_,
        "Classification Report": classification_report(y_val, y_val_pred_rf, output_dict=True),
        "Quadratic Weighted Kappa": rf_kappa,
    }
    print(f"Best Parameters (Random Forest): {grid_search_rf.best_params_}")
    print(classification_report(y_val, y_val_pred_rf))
    print(f"Quadratic Weighted Kappa: {rf_kappa:.4f}")

    if rf_kappa > best_kappa:
        best_kappa = rf_kappa
        best_model = best_rf

    """
    # Model 2: Logistic Regression
    print("\n--- Training Logistic Regression ---")
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2']
    }
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=3, scoring=scorer)
    grid_search_lr.fit(X_train_scaled, y_train)

    best_lr = grid_search_lr.best_estimator_
    y_val_pred_lr = best_lr.predict(X_val_scaled)
    lr_kappa = quadratic_weighted_kappa(y_val, y_val_pred_lr)
    results["Logistic Regression"] = {
        "Best Parameters": grid_search_lr.best_params_,
        "Classification Report": classification_report(y_val, y_val_pred_lr, output_dict=True),
        "Quadratic Weighted Kappa": lr_kappa,
    }
    print(f"Best Parameters (Logistic Regression): {grid_search_lr.best_params_}")
    print(classification_report(y_val, y_val_pred_lr))
    print(f"Quadratic Weighted Kappa: {lr_kappa:.4f}")

    if lr_kappa > best_kappa:
        best_kappa = lr_kappa
        best_model = best_lr

    # Model 3: SVM
    print("\n--- Training SVM ---")
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    svm = SVC(class_weight="balanced", random_state=42)
    grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=3, scoring=scorer)
    grid_search_svm.fit(X_train_scaled, y_train)

    best_svm = grid_search_svm.best_estimator_
    y_val_pred_svm = best_svm.predict(X_val_scaled)
    svm_kappa = quadratic_weighted_kappa(y_val, y_val_pred_svm)
    results["SVM"] = {
        "Best Parameters": grid_search_svm.best_params_,
        "Classification Report": classification_report(y_val, y_val_pred_svm, output_dict=True),
        "Quadratic Weighted Kappa": svm_kappa,
    }
    print(f"Best Parameters (SVM): {grid_search_svm.best_params_}")
    print(classification_report(y_val, y_val_pred_svm))
    print(f"Quadratic Weighted Kappa: {svm_kappa:.4f}")

    if svm_kappa > best_kappa:
        best_kappa = svm_kappa
        best_model = best_svm

    # Model 4: Neural Network (no grid search implemented here)
    print("\n--- Training Neural Network ---")
    y_train = np.array(y_train, dtype=int)
    y_val = np.array(y_val, dtype=int)

    num_classes = len(np.unique(y_train))
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    input_size = X_train.shape[1]
    model_nn = NeuralNetwork(input_size, num_classes)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model_nn.parameters(), lr=0.0001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model_nn.train()
        optimizer.zero_grad()
        outputs = model_nn(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model_nn.eval()
        with torch.no_grad():
            val_outputs = model_nn(X_val_tensor)
            val_predictions = torch.argmax(val_outputs, dim=1)

    nn_kappa = quadratic_weighted_kappa(y_val, val_predictions.numpy())
    results["Neural Network"] = {
        "Classification Report": classification_report(y_val, val_predictions.numpy(), output_dict=True),
        "Quadratic Weighted Kappa": nn_kappa,
    }
    print(classification_report(y_val, val_predictions.numpy()))
    print(f"Quadratic Weighted Kappa: {nn_kappa:.4f}")

    if nn_kappa > best_kappa:
        best_kappa = nn_kappa
        best_model = model_nn

    print("\n--- Best Model Selected ---")
    print(f"Best Model: {type(best_model).__name__}")
    print(f"Best Quadratic Weighted Kappa: {best_kappa:.4f}")
    """
    return results, best_model
