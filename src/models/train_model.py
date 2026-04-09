import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib

print("Loading data...")

df = pd.read_csv("data/equilend_mock_data.csv")

print("Data loaded:", df.shape)
print("Columns are:", df.columns)

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Convert categorical → numeric
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

# 🔥 FIX employment_length (string → number)
df["employment_length"] = df["employment_length"].str.extract('(\d+)').astype(float)

# Target column
target = "default_status"

# Features & labels
X = df.drop(columns=[target])
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

print("Model trained!")

# Predict
y_pred = model.predict_proba(X_test)[:, 1]

# Evaluate
auc = roc_auc_score(y_test, y_pred)
print("AUC Score:", auc)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")