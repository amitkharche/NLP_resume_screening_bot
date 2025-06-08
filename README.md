# 🧠 Resume Screening Bot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

## 📌 Business Use Case

Recruiters often spend hours manually screening resumes. This Resume Screening Bot uses NLP to automatically classify resumes based on the most suitable job category, helping save time and improve efficiency in hiring processes.

## 🧠 Features Used

- Resume text is converted into TF-IDF vectors.
- Model used: Logistic Regression Classifier.

## ⚙️ Pipeline Steps

### Training (`model_training.py`)
- Load simulated resume data
- Preprocess resumes using TF-IDF
- Train Logistic Regression classifier
- Evaluate model and save it

### App (`app.py`)
- Upload CSV of resumes
- Predict job fit using trained model
- Display and download results

## 🚀 Usage Instructions

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the model
```bash
python model_training.py
```

### Step 3: Run the Streamlit app
```bash
streamlit run app.py
```

## 🗂 Project Structure
```
resume_screening_bot_project/
├── data/                         # Synthetic dataset
├── model/                        # Trained model
├── app.py                        # Streamlit app
├── model_training.py             # Training pipeline
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── .github/                      # GitHub templates & CI
```

## 📄 License
This project is licensed under the MIT License.
