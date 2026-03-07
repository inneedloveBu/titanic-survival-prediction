
# Titanic Survival Prediction – An End-to-End Data Science Project

[![bilibili](https://img.shields.io/badge/🎥-View%20on%20Bilibili-red)](https://www.bilibili.com/video/BV1tFdWY4Epd)  

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Huggingon%20Faceon%20Spaces-blue)](https://huggingface.co/spaces/indeedlove/titanic-survival-predictor) 
[![GitHub](https://img.shields.io/badge/📂-GitHub-black)](https://github.com/inneedloveBu/titanic-survival-prediction)


---

## Project Overview

This project investigates the application of machine learning techniques for predicting passenger survival on the Titanic dataset, originally introduced in the Kaggle competition.

The study implements a complete **end-to-end data science pipeline**, including:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Hyperparameter Optimization
- Model Evaluation
- Interactive Model Deployment

A **Random Forest classifier** is used as the primary model. Its performance is improved through **systematic feature engineering and hyperparameter tuning using GridSearchCV with cross-validation**.

To demonstrate practical deployment capabilities, the trained model is integrated into an **interactive web application built with Gradio**, allowing real-time survival prediction based on passenger attributes.

---

<img width="1360" height="633" alt="ScreenShot" src="https://github.com/user-attachments/assets/a786702c-f636-4c35-b2c6-1e69a29075bb" />




## Motivation

This project was developed to strengthen practical skills in **applied machine learning and data science engineering**.  

It aims to demonstrate the ability to build a **reproducible data science workflow**, from raw data exploration to a deployable predictive system.

The project reflects core competencies required in modern **Data Science and Artificial Intelligence MSc programs**, including statistical analysis, machine learning modelling, and data-driven application development.

---

## Technical Skills Demonstrated

### Data Processing
- Data cleaning and missing value handling using **pandas**
- Encoding categorical variables
- Feature scaling and preprocessing pipelines using **Scikit-learn**

### Exploratory Data Analysis (EDA)
- Statistical data exploration
- Visualization of feature relationships using **matplotlib** and **seaborn**
- Identification of influential predictors

### Feature Engineering
Creation of domain-informed predictive features such as:

- **FamilySize**
- **IsAlone**
- **Title extraction from passenger names**

These engineered variables improve model performance and interpretability.

### Machine Learning Modeling
- Binary classification using **RandomForestClassifier**
- Model training and validation
- Feature importance analysis

### Model Optimization
- Hyperparameter tuning using **GridSearchCV**
- **k-fold cross-validation** to estimate generalization performance

### Model Deployment
- Model persistence using **joblib**
- Interactive web application built with **Gradio**

### Reproducibility & Engineering
- Version control using **Git**
- Modular project structure
- Fully documented execution pipeline

---

## Project Structure


titanic-survival-prediction
│
├── data
│ ├── train.csv
│ └── test.csv
│
├── notebooks
│ └── eda_analysis.ipynb
│
├── model_training.py
├── app_gradio.py
├── requirements.txt
└── README.md


This structure separates **data analysis, model training, and deployment**, reflecting common industry and research project organization.

---

## System Demonstration

The trained model is deployed through a **Gradio web interface** that allows users to input passenger attributes and obtain a survival prediction.

![ScreenShot](https://github.com/user-attachments/assets/a786702c-f636-4c35-b2c6-1e69a29075bb)

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/inneedloveBu/titanic-survival-prediction.git
cd titanic-survival-prediction
2. Create a virtual environment
python -m venv venv

Activate environment

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Download Kaggle Dataset

Download:

train.csv

test.csv

from the Kaggle competition page:

https://www.kaggle.com/c/titanic/data

Place them in the project root directory.

5. Train the model
python model_training.py

This will:

train the Random Forest model

perform hyperparameter tuning

output evaluation metrics

generate a prediction file

6. Launch the interactive application
python app_gradio.py

Then open:

http://localhost:7860
Key Results & Insights

The baseline Random Forest model achieved a validation accuracy of:

0.8156

After feature engineering and hyperparameter optimization using GridSearchCV:

Best Cross-Validation Score: 0.8258

Validation Accuracy: 0.8101

Feature importance analysis indicates that the most influential predictors are:

Sex

Passenger Class (Pclass)

Fare

These findings align with historical accounts of evacuation priorities during the Titanic disaster.

Future Work

Potential improvements include:

applying Gradient Boosting models such as XGBoost or LightGBM

experimenting with ensemble learning techniques

implementing feature selection methods

improving model interpretability using SHAP values

containerizing the project using Docker

deploying the application to a public cloud environment

License

This project is licensed under the MIT License.
