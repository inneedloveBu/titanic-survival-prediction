# Titanic Survival Prediction - A Data Science Project

## Project Overview
This project implements a machine learning solution to predict passenger survival on the Titanic, based on the famous [Kaggle competition](https://www.kaggle.com/c/titanic). It demonstrates a complete data science pipeline: from **Exploratory Data Analysis (EDA)** and **feature engineering** to **model training/optimization** with `GridSearchCV`, **model persistence**, and finally, the deployment of an **interactive web application** using Gradio.

The project is structured to reflect industry-relevant practices and aligns closely with core modules of advanced Data Science curricula, such as Applied Statistics, Machine Learning, and Big Data Processing.

## Motivation
This project was developed to solidify my practical skills in applied data science and to build a portfolio piece that showcases end-to-end competency—from data wrangling to a deployable product. It directly demonstrates the skills required for data-intensive MSc programs.

## Technical Skills Demonstrated
*   **Data Wrangling**: Handling missing values, encoding categorical variables, feature scaling using `sklearn` pipelines.
*   **Exploratory Data Analysis (EDA)**: Utilizing `pandas`, `matplotlib`, and `seaborn` to uncover data patterns and relationships.
*   **Feature Engineering**: Creating new predictive features (e.g., `FamilySize`, `Title`, `IsAlone`) from raw data.
*   **Machine Learning Modeling**: Building, evaluating, and interpreting a `RandomForestClassifier`.
*   **Model Optimization**: Performing hyperparameter tuning using `GridSearchCV` with cross-validation.
*   **Model Deployment**: Building an interactive web interface with `Gradio` and persisting the model using `joblib`.
*   **Version Control & Reproducibility**: Managing code with `Git` and ensuring the project is fully documented and runnable.

## Project Structure
<img width="740" height="288" alt="ScreenShot_2025-12-22_194001_573" src="https://github.com/user-attachments/assets/da3758b0-5f6e-4adb-9dc4-217854d35834" />

## screen picture
![微信图片_20251222193524_96_116](https://github.com/user-attachments/assets/93c33670-5bbe-4719-a62b-859e6fec09d2)



## How to Run the Project
1.  **Clone the repository**
    ```bash
    git clone https://github.com/inneedloveBu/titanic-survival-prediction.git
    cd titanic-survival-prediction
    ```
2.  **Set up a virtual environment and install dependencies**
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Download the data**
    *   Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/titanic/data).
    *   Place them in the project root directory.
4.  **Train the model and generate predictions**
    ```bash
    python model_training.py
    ```
    This will output model performance metrics and create a submission file.
5.  **Launch the interactive web application**
    ```bash
    python app_gradio.py
    ```
    Then open your browser and go to `http://localhost:7860`.

## Key Results & Insights
*   The baseline Random Forest model achieved a validation accuracy of **XX.XX%**.
*   After feature engineering and hyperparameter tuning via `GridSearchCV`, the optimized model achieved:
    *   **Best Cross-Validation Score:** **0.8258**
    *   **Validation Set Accuracy:** **0.8101**
*   The most important features for predicting survival, according to the model, were **Sex, Passenger Class (Pclass), and Fare**.

## Next Steps & Potential Improvements
*   Experiment with more advanced algorithms (e.g., Gradient Boosting with `XGBoost` or `LightGBM`).
*   Perform deeper feature engineering and selection.
*   Containerize the application using Docker for easier deployment.
*   Deploy the Gradio app to a cloud platform like [Hugging Face Spaces](https://huggingface.co/spaces) for public access.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
