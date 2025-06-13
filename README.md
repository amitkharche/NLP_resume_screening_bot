
---

# Resume Fit Verifier Bot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

## Business Use Case

Hiring teams often face the challenge of manually scanning hundreds of resumes for each job posting. This Resume Fit Verifier automates the screening process using NLP and machine learning, classifying whether a resume is a **Fit** or **Unfit** for a given **Job Title**.

The tool is designed to reduce recruiter effort, speed up shortlisting, and introduce consistency in candidate evaluation.

---

## Features

* ✅ Resume text and job title are combined as contextual input.
* ✅ Labels are heuristically derived using token overlap.
* ✅ Model uses TF-IDF feature extraction and Logistic Regression.
* ✅ Built-in visualization for model evaluation and predictions.
* ✅ Streamlit UI for interactive predictions and CSV downloads.

---

## Model Workflow

### Step-by-Step Pipeline

#### `model_training.py`

* Loads resume-job title dataset (`Resume_Text`, `Job_Title`).
* Uses token-overlap to label resumes as `"Fit"` or `"Unfit"`.
* Concatenates `Resume_Text + [SEP] + Job_Title` to form input.
* TF-IDF transforms input into numeric feature vectors.
* Logistic Regression classifies input into binary output.
* Saves model (`job_fit_model.pkl`) and labeled CSV.

#### `app.py`

* Lets you upload a CSV with `Resume_Text` and `Job_Title`.
* Predicts job fit with trained model.
* Shows top predictions and offers CSV download.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/amitkharche/NLP_resume_screening_bot.git
cd NLP_resume_screening_bot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python model_training.py
```

* This will generate:

  * `model/job_fit_model.pkl`
  * `data/resume_job_pairs_labeled.csv`

### Launch the Streamlit App

```bash
streamlit run app.py
```

* Upload a CSV with these two columns:

  * `Resume_Text` — the full resume content
  * `Job_Title` — the job title being applied for

* The model will classify each row as **Fit** or **Unfit**

---

## Project Structure

```
resume-fit-verifier/
├── data/
│   ├── resume_job_pairs.csv           # Input resumes with job titles
│   └── resume_job_pairs_labeled.csv   # Labeled output (Fit/Unfit)
├── model/
│   └── job_fit_model.pkl              # Trained logistic regression model
├── app.py                             # Streamlit user interface
├── model_training.py                  # Full training pipeline
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .github/                           # GitHub actions, templates, etc. (optional)
```

---

## Evaluation Logic

* A confusion matrix is plotted using test predictions
* Labels are handled dynamically with `unique_labels()` to avoid errors if one class is missing in the split

---

## Sample CSV Format

```csv
Resume_Text,Job_Title
"Experienced Python developer with ML exposure","Machine Learning Engineer"
"Managed data pipelines and dashboards","Data Analyst"
```

---

## License

This project is licensed under the **MIT License** — feel free to use, modify, and share.

---

## Let's Connect

Have questions or want to collaborate? Reach out here:

* [LinkedIn](https://www.linkedin.com/in/amit-kharche)
* [Medium](https://medium.com/@amitkharche14)
* [GitHub](https://github.com/amitkharche)

---