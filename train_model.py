# train_model.py
from model_utils import load_dataset, train_random_forest, evaluate_models, save_best_model, save_scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("🔎 Loading dataset...")
X, y = load_dataset()

print(f"📦 Loaded {len(X)} samples.")

print("⚙️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Train Random Forest model only
models = {}
rf = train_random_forest(X_train, y_train)
models["RandomForest"] = rf

# Evaluate
results = evaluate_models(models, X_test, y_test)

# Save best
save_best_model(results, X_test, y_test)
save_scaler(scaler)
