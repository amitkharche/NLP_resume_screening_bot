"""
Streamlit app to verify job fit from resumes using a trained binary classification model.
Input: Resume_Text + Job_Title
Output: Fit or Unfit
"""

import streamlit as st
import pandas as pd
import pickle

# ── Page Config ────────────────────────────────────────────
st.set_page_config(page_title="🧠 Resume Fit Verifier", layout="wide")
st.title("🧠 Resume Fit Verifier")

# ── Load Trained Model ─────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model/job_fit_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ── Predict Function ───────────────────────────────────────
def predict_job_fit(df, model):
    df["input_text"] = df["Resume_Text"] + " [SEP] " + df["Job_Title"]
    df["Job_Fit"] = model.predict(df["input_text"])
    return df

# ── Main App Logic ─────────────────────────────────────────
def main():
    uploaded_file = st.file_uploader("📤 Upload a CSV with 'Resume_Text' and 'Job_Title' columns", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        expected_cols = {"Resume_Text", "Job_Title"}
        
        if not expected_cols.issubset(df.columns):
            st.error(f"❌ Uploaded file must contain columns: {', '.join(expected_cols)}")
            return
        
        st.subheader("📄 Uploaded Data Sample")
        st.dataframe(df.head())

        model = load_model()
        if model:
            st.info("🔍 Running prediction...")
            result_df = predict_job_fit(df.copy(), model)

            st.subheader("✅ Job Fit Prediction")
            st.dataframe(result_df[["Resume_Text", "Job_Title", "Job_Fit"]].head())

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results as CSV", csv, "job_fit_predictions.csv", "text/csv")

# ── Run App ────────────────────────────────────────────────
if __name__ == "__main__":
    main()
