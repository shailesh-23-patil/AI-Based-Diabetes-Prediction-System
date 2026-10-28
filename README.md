# 🩺 AI-Based Diabetes Prediction System

## 📌 Project Overview

The **AI-Based Diabetes Prediction System** is a machine learning project that predicts whether a person is likely to be classified as diabetic based on selected health, lifestyle, and medical features.

The project uses a **Random Forest Classifier** and provides a user-friendly web interface built with Streamlit. Users can enter health-related information and receive a predicted class along with the model-estimated probability.

> **Disclaimer:** This project is developed for educational purposes. Its predictions are not a medical diagnosis and should not be used to make healthcare decisions.

## 🎯 Project Objectives

- Develop a supervised machine learning model for diabetes classification.
- Prepare and preprocess health-related data.
- Train a Random Forest classification model.
- Evaluate the model using standard classification metrics.
- Build an interactive web application using Streamlit.
- Display predictions and estimated probabilities.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Storage:** Joblib
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook
- **Dataset Format:** CSV

## 📂 Project Structure

```text
Diabetes-Prediction-System/
│
├── app.py
├── diabetes_prediction.ipynb
├── evaluation_results.csv
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── diabetes_dataset.csv
│
├── models/
│   ├── diabetes_model.pkl
│   └── diabetes_preprocessor.pkl
│
└── docs/
    ├── Diabetes_Prediction_Project_Report.pdf
    └── Diabetes_Prediction_Project_Report.docx
```

## 📊 Dataset Information

**Source:** *Diabetes Health Indicators Dataset* on Kaggle
(https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset)

The project uses a diabetes dataset containing **100,000 records and 31 columns**. According to its Kaggle description, the data is **synthetic** (generated to resemble real medical data), so it does not contain records of real patients.

The dataset includes health, demographic, lifestyle, and medical-related features, such as:

- Age and gender
- BMI and waist-to-hip ratio
- Blood pressure
- Cholesterol levels
- Fasting and postprandial glucose
- HbA1c and insulin level
- Physical activity and lifestyle information
- Family history of diabetes

**Target variable:** `diagnosed_diabetes`

- `0` — Non-diabetic class (about 40% of the records)
- `1` — Diabetic class (about 60% of the records)

There are no missing values in the dataset. The columns `diabetes_stage` and `diabetes_risk_score` were excluded from the model inputs to reduce target leakage, which leaves **28 input features**.

## ⚙️ Data Preprocessing

The following steps were performed:

1. Loaded the dataset using Pandas.
2. Checked for missing values and duplicate records.
3. Separated the input features from the target variable.
4. Excluded columns that could cause target leakage.
5. Applied One-Hot Encoding to categorical features.
6. Split the data into training and testing sets using an 80:20 ratio.
7. Used stratified sampling to preserve the target class distribution.

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier**, an ensemble learning algorithm that combines multiple decision trees to make predictions.

Model configuration:

- Number of trees: 100
- Random state: 42
- Training/testing split: 80% / 20%
- Training records: 80,000
- Testing records: 20,000

The trained model and preprocessing pipeline are saved in the `models/` directory.

## 📈 Model Evaluation Results

The model was evaluated on the held-out test dataset (20,000 records).

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 91.96% |
| Precision (diabetic class) | 99.94% |
| Recall (diabetic class) | 86.65% |
| F1-Score (diabetic class) | 92.82% |
| ROC-AUC | 94.13% |

**Confusion matrix:** 7,994 true negatives, 6 false positives, 1,602 false negatives, 10,398 true positives.

**Most important features:** HbA1c, postprandial glucose and fasting glucose together account for roughly 70% of the model's feature importance.

These results describe performance against the dataset's labels. They do not establish clinical accuracy or suitability for real-world diagnosis.

## 💻 Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible Python version
- pip
- A code editor such as VS Code (optional)
- A web browser

### Step 1: Get the project

Unzip the submitted project folder and open it in your terminal:

```bash
cd Diabetes-Prediction-System
```

(If you received the project through a Git repository, clone it first with `git clone <repository-url>`.)

### Step 2: Create a virtual environment

**Windows:**

```bash
python -m venv .venv
```

### Step 3: Activate the environment

For Command Prompt:

```cmd
.venv\Scripts\activate
```

For PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install the required libraries

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Check the project files

Make sure the following files are available:

- `data/diabetes_dataset.csv`
- `models/diabetes_model.pkl`
- `models/diabetes_preprocessor.pkl`
- `app.py`

The web application requires the saved model and preprocessor files. If they are missing, open `diabetes_prediction.ipynb` and run all cells to generate them again.

### Step 6: Run the application

From the main project directory, execute:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your web browser.

### Optional: Run the notebook

```bash
python -m pip install notebook
jupyter notebook
```

Open `diabetes_prediction.ipynb` and choose **Kernel → Restart & Run All**.

## 🖥️ How to Use the Web Application

1. Start the application using the command above.
2. Open the local URL displayed in your terminal.
3. Go through the five tabs (Personal, Lifestyle, Medical History, Body, Cholesterol & Glucose) and enter the requested information. Typical values are pre-filled.
4. On the last tab, press the **Predict Diabetes Label** button.
5. View the predicted class, the model-estimated probability and the summary of the submitted values.
6. Use the result for educational demonstration only, not for medical decisions.

## 📁 Important Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `diabetes_prediction.ipynb` | Data analysis, preprocessing, model training, and evaluation |
| `evaluation_results.csv` | Saved evaluation metrics |
| `requirements.txt` | Python packages needed to run the project |
| `.streamlit/config.toml` | Streamlit theme settings for the web app |
| `diabetes_model.pkl` | Trained Random Forest model |
| `diabetes_preprocessor.pkl` | Saved feature preprocessing pipeline |
| `diabetes_dataset.csv` | Dataset used for training and testing |
| `docs/Diabetes_Prediction_Project_Report` | Project report (PDF and Word) |

## ⚠️ Limitations

- The dataset is synthetic, so performance may not generalize to real patients or other populations.
- The model relies heavily on HbA1c and glucose values, which are themselves used to diagnose diabetes.
- About 13% of diabetic records in the test set were missed (1,602 of 12,000).
- Model-estimated probabilities are not necessarily calibrated clinical risks.
- The system has not been established as a clinically validated diagnostic tool.
- A healthcare professional should assess any real medical concerns.

## 🚀 Future Enhancements

- Improve recall, for example by lowering the decision threshold or using class weights.
- Train a version without HbA1c and glucose inputs to study lifestyle-based prediction.
- Compare Random Forest with other classification algorithms.
- Add interactive visualizations and feature explanations.
- Investigate probability calibration and independent validation.

## 👨‍🎓 Project Information

**Project Title:** AI-Based Diabetes Prediction System

**Author:** Shailesh Rajendra Patil

**Project Type:** Academic Machine Learning Project

**Application Framework:** Streamlit

**Model:** Random Forest Classifier

---

*Developed for academic learning and demonstration purposes.*