# train_pipeline.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load insurance dataset
df = pd.read_csv("insurance_data.csv")

# Columns for numeric and categorical features
numeric_features = ["AGE", "PREMIUM_AMOUNT", "CLAIM_AMOUNT", "INCIDENT_HOUR_OF_THE_DAY"]
categorical_features = ["AGENT_ID", "VENDOR_ID", "INSURANCE_TYPE"]

X = df[numeric_features + categorical_features]
y = df["CLAIM_STATUS"]  # target variable

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Full pipeline with classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
pipeline.fit(X, y)

# Save pipeline to disk
joblib.dump(pipeline, "epic5_model_pipeline.pkl")
print("Pipeline trained and saved successfully!")