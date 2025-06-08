"""
Model training script for resume screening bot using Logistic Regression.
"""

import os
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Config
DATA_PATH = "data/synthetic_resumes.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "resume_model.pkl")

def load_data():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded data with shape: {df.shape}")
    return df

def train_model(df):
    X = df["Resume_Text"]
    y = df["Job_Fit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return pipeline

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_PATH}")

def main():
    df = load_data()
    model = train_model(df)
    save_model(model)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
