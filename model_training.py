import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from utils import load_data, clean_replace_zero_with_median, FEATURES, TARGET

def train_and_save(path='data/diabetes.csv', model_out='model.pkl', scaler_out='scaler.pkl'):
    df = load_data(path)
    df = clean_replace_zero_with_median(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'logreg': LogisticRegression(max_iter=500, random_state=42),
        'rf': RandomForestClassifier(n_estimators=200, random_state=42)
    }

    best_model = None
    best_auc = 0

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:,1]
        auc = roc_auc_score(y_test, probs)
        print(f"Model: {name}, AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = model

    print('\nBest model:', best_model)
    joblib.dump(best_model, model_out)
    joblib.dump(scaler, scaler_out)

    preds = best_model.predict(X_test_s)
    print('\nAccuracy:', accuracy_score(y_test, preds))
    print('\nClassification report:\n', classification_report(y_test, preds))

if __name__ == '__main__':
    train_and_save()
