"""
Streamlit app to predict job fit from resumes using a trained classification model.
"""

import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="🧠 Resume Screening Bot", layout="wide")
st.title("🧠 Resume Screening Bot")

@st.cache_resource
def load_model():
    try:
        with open("model/resume_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_job_fit(df, model):
    df["Predicted_Job_Fit"] = model.predict(df["Resume_Text"])
    return df

def main():
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Resume_Text' column", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Resume_Text" not in df.columns:
            st.error("The uploaded file must have a 'Resume_Text' column.")
            return

        st.write("### Uploaded Resumes", df.head())

        model = load_model()
        if model:
            result_df = predict_job_fit(df, model)
            st.write("### Predicted Job Fit", result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions as CSV", csv, "predicted_job_fit.csv", "text/csv")

if __name__ == "__main__":
    main()
