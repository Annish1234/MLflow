import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. base_dir is the folder containing train.py (i.e., model/)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Load data
df = pd.read_csv(os.path.join(base_dir, "creditcard_10k_balanced.csv"))
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X.applymap(lambda x: str(x).replace(" ", ""))  
X = X.apply(pd.to_numeric, errors='coerce') 

X = X.fillna(X.mean()) 

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Save scaler directly in base_dir (model/)
joblib.dump(scaler, os.path.join(base_dir, "scaler.pkl"))

# 6. Train & evaluate
best_model = None
best_score = -1
best_name = None

for name, model in {
    "SVM": SVC(probability=True),
    "LightGBM": LGBMClassifier(class_weight="balanced"),
}.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    # Display only accuracy
    print(f"{name} accuracy: {acc:.4f}")

    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

# 7. Save best model directly in base_dir (model/)
if best_model:
    joblib.dump(best_model, os.path.join(base_dir, "best_model.pkl"))
    print(f"Best model accuracy: {best_score:.4f}")
