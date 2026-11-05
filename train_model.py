import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/student-mat.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Basic cleaning
df = df.drop_duplicates()
df = df.fillna(df.mode().iloc[0])

# 3. Create academic-risk target
def get_risk_level(grade):
    if grade >= 15:
        return "Low"
    elif grade >= 10:
        return "Medium"
    else:
        return "High"

df["risk_level"] = df["G3"].apply(get_risk_level)

# 4. Feature engineering
df["attendance_score"] = 1 - (df["absences"] / df["absences"].max())
df["failure_impact"] = df["failures"] * df["absences"]
df["study_support_score"] = df["studytime"] + df["famsup"].map({"yes": 1, "no": 0})

# 5. Select input features
# Exclude G1 and G2 to make an early-warning prediction system
X = df.drop(columns=["G1", "G2", "G3", "risk_level"])
y_grade = df["G3"]
y_risk = df["risk_level"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 6. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 7. Train-test split
X_train, X_test, y_grade_train, y_grade_test, y_risk_train, y_risk_test = train_test_split(
    X,
    y_grade,
    y_risk,
    test_size=0.2,
    random_state=42,
    stratify=y_risk
)

# 8. Final grade prediction model
grade_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    ))
])

grade_model.fit(X_train, y_grade_train)
grade_predictions = grade_model.predict(X_test)

mae = mean_absolute_error(y_grade_test, grade_predictions)
rmse = np.sqrt(mean_squared_error(y_grade_test, grade_predictions))
r2 = r2_score(y_grade_test, grade_predictions)

print("\n==============================")
print("GRADE PREDICTION RESULTS")
print("==============================")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 9. Academic-risk classification model
risk_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ))
])

risk_model.fit(X_train, y_risk_train)
risk_predictions = risk_model.predict(X_test)

print("\n==============================")
print("ACADEMIC RISK CLASSIFICATION RESULTS")
print("==============================")
print("Accuracy:", round(accuracy_score(y_risk_test, risk_predictions), 3))
print("\nClassification Report:")
print(classification_report(y_risk_test, risk_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_risk_test, risk_predictions))
# 10. Cross-validation
cv_scores = cross_val_score(
    risk_model,
    X,
    y_risk,
    cv=5,
    scoring="f1_weighted"
)

print("\n5-Fold Cross-Validation Weighted F1 Scores:")
print(cv_scores)
print("Average Weighted F1 Score:", round(cv_scores.mean(), 3))

# 11. Save trained models
joblib.dump(grade_model, os.path.join(MODEL_DIR, "grade_model.joblib"))
joblib.dump(risk_model, os.path.join(MODEL_DIR, "risk_model.joblib"))

print("\nModels saved successfully:")
print("models/grade_model.joblib")
print("models/risk_model.joblib")