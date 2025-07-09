# train.py (Updated for all platforms)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load data
df = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")

# Clean follower count
def parse_followers(value):
    if isinstance(value, str):
        value = value.strip("~+").upper().replace(",", "").strip()
        try:
            if 'M' in value:
                return float(value.replace('M', '')) * 1_000_000
            elif 'K' in value:
                return float(value.replace('K', '')) * 1_000
            else:
                return float(value)
        except:
            return np.nan
    return value

df["Follower Count"] = df["Follower Count"].apply(parse_followers)

# Clean engagement rate
def parse_engagement_rate(rate):
    if isinstance(rate, str):
        rate = rate.strip("~%").strip()
        try:
            return float(rate)
        except:
            return np.nan
    return rate

df["Engagement Rate"] = df["Engagement Rate"].apply(parse_engagement_rate)

df = df.dropna(subset=["Follower Count", "Engagement Rate", "Platform Used", "Category", "Country"])

# Generate simulated score (optional)
def generate_score(row):
    score = (row["Engagement Rate"] * 10) + (np.log1p(row["Follower Count"]) * 3)
    return min(score, 100)

df["Score"] = df.apply(generate_score, axis=1)

# Features
features = ["Follower Count", "Engagement Rate", "Platform Used", "Category", "Country"]
X = df[features]
y = df["Score"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ["Follower Count", "Engagement Rate"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Platform Used", "Category", "Country"])
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, "influencer_model.pkl")
print("Model saved successfully as influencer_model.pkl")
