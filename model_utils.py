# model_utils.py
import os
import numpy as np
from features_extraction import extract_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time

def load_dataset(data_path="data"):
    X, y = [], []
    for label in ['male', 'female']:
        folder = os.path.join(data_path, label)
        for file in os.listdir(folder):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(folder, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(0 if label == 'male' else 1)
    return np.array(X), np.array(y)

def train_models(X_train, y_train):
    svm = SVC(kernel='rbf', probability=True)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    print("🔁 Training SVM...")
    svm.fit(X_train, y_train)
    print("🔁 Training Random Forest...")
    rf.fit(X_train, y_train)

    return svm, rf

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n📊 {name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))
        results[name] = (model, acc, y_pred)
    return results

def save_best_model(models_results, X_test, y_test):
    best_model_name = max(models_results, key=lambda x: models_results[x][1])
    best_model, best_acc, y_pred = models_results[best_model_name]

    print(f"\n✅ Best model: {best_model_name} with accuracy {best_acc:.4f}")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
    plt.title(f"{best_model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"💾 Saved best model as model.pkl")

def save_scaler(scaler):
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("💾 Saved scaler as scaler.pkl")


from sklearn.ensemble import RandomForestClassifier



def train_random_forest(X_train, y_train):
    print("🌲 Training Random Forest...")
    start = time.time()

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)

    end = time.time()
    print(f"✅ Training completed in {end - start:.2f} seconds.")
    return rf

