# 🤖 AI Job Market — Salary Predictor

A complete end-to-end Data Science project: data cleaning,
exploratory analysis and Machine Learning modeling to predict
salaries in the Data & AI job market.

---

## 📌 Objective
Predict the annual salary (in USD) of a Data & AI job position
based on the job characteristics and the candidate's profile.

---

## 📁 Project Structure
ai-job-salary-predictor/
├── notebooks/
│   ├── 01_Cleaning_.ipynb        # Data cleaning and preparation
│   ├── 02_visualisation.ipynb    # Exploratory analysis and charts
│   └── 03_machine_learning.ipynb # Modeling and model evaluation
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Cleaned dataset
├── models/
│   └── model_salary.pkl          # Saved model
├── figures/                      # Generated charts
├── app/
│   └── app.py                    # Streamlit interface
└── requirements.txt

---

## 🔧 Technologies Used
- **Python** — main language
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — data visualization
- **Scikit-learn** — Machine Learning
- **Streamlit** — user interface

---

## 🤖 Models Tested
| Model | MAE (USD) | R² |
|---|---|---|
| Linear Regression ✅ | ~16,200 | best |
| Decision Tree | - | - |
| Random Forest | - | - |

The best performing model is **Linear Regression**.

---

## 📊 Dataset
Synthetic dataset of 10,000+ job postings in Data & AI fields
(2020–2026), covering 7 countries and 6 job titles.

---

