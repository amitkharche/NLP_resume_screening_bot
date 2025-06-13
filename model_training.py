"""
Model-training script for binary job-fit prediction.

Input  CSV : Resume_Text, Job_Title
Output CSV : Resume_Text, Job_Title, Job_Fit (Fit / Unfit)

Label logic: If any word from Job_Title is found in Resume_Text → Fit
Model: TF-IDF + Logistic Regression
"""

import os
import re
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ── Config ─────────────────────────────────────────────────────────────
RAW_DATA   = "data/resume_job_pairs.csv"
AUG_DATA   = "data/resume_job_pairs_labeled.csv"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "job_fit_model.pkl")

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s • %(levelname)s • %(message)s"
)
log = logging.getLogger(__name__)

# ── 1. Load raw data safely ────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA)
    df.columns = df.columns.str.strip().str.replace('\u200b', '').str.replace('\xa0', '')
    
    required_cols = {"Resume_Text", "Job_Title"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}. Found: {list(df.columns)}")
    
    log.info(f"Loaded raw data: {df.shape[0]} rows")
    return df

# ── 2. Token-overlap labeling logic ────────────────────────────────────
def add_job_fit_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule: If any token from job title exists in resume → Fit else Unfit
    """
    def is_fit(row):
        title_tokens = set(re.findall(r"\w+", row["Job_Title"].lower()))
        resume_tokens = set(re.findall(r"\w+", row["Resume_Text"].lower()))
        return "Fit" if title_tokens & resume_tokens else "Unfit"
    
    df["Job_Fit"] = df.apply(is_fit, axis=1)
    log.info("Heuristic labels generated.\nDistribution:\n%s", df["Job_Fit"].value_counts())
    return df

# ── 3. Train binary classification model ───────────────────────────────
def train_model(df: pd.DataFrame):
    df["input_text"] = df["Resume_Text"] + " [SEP] " + df["Job_Title"]
    X = df["input_text"]
    y = df["Job_Fit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])

    log.info("Training model...")
    pipeline.fit(X_train, y_train)

    log.info("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    log.info("\n%s", classification_report(y_test, y_pred))

    return pipeline

# ── 4. Save model and labeled data ─────────────────────────────────────
def save_outputs(model, df_labeled):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"✅ Model saved to: {MODEL_PATH}")

    df_labeled.to_csv(AUG_DATA, index=False)
    log.info(f"📁 Labeled CSV written to: {AUG_DATA}")

# ── Main Entrypoint ────────────────────────────────────────────────────
def main():
    raw_df = load_raw()
    labeled_df = add_job_fit_label(raw_df.copy())
    model = train_model(labeled_df)
    save_outputs(model, labeled_df)
    log.info("🎉 Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
