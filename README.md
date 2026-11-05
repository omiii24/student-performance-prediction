# 🎓 Student Performance Prediction System

🔗 **Live Demo:** https://student-performance-prediction-dwpguteaki3iivm9xa9fim.streamlit.app/

A machine learning web app that predicts a student's final grade (out of 20) and academic risk level (Low / Medium / High) from study habits, attendance, family background and school data. Built with Python, scikit-learn and Streamlit.

## Features

- Predicts the final grade (G3) using a regression model
- Classifies academic risk as Low, Medium or High, with class probabilities
- Interactive Streamlit interface with 30 input features
- Engineered features: attendance score, failure impact, study support score
- Gives recommendations based on the predicted risk level
- Early-prediction design: G1 and G2 are excluded to avoid target leakage

## Tech Stack

Python, Pandas, scikit-learn, Joblib, Streamlit

## Dataset

UCI Student Performance dataset (`student-mat.csv`), 395 students, 30 input attributes covering demographics, family, school and lifestyle.

## Model Results

| Model | Metric | Score |
|-------|--------|-------|
| Grade model (Random Forest Regressor) | MAE | 3.21 |
| Grade model (Random Forest Regressor) | RMSE | 4.22 |
| Grade model (Random Forest Regressor) | R² | 0.12 |
| Risk model (Random Forest Classifier) | Accuracy | 0.52 |
| Risk model (Random Forest Classifier) | 5-fold CV weighted F1 | 0.50 |

Evaluated on a stratified 80/20 train-test split (79 test students). Risk levels are derived from the final grade: Low (≥ 15), Medium (10-14), High (< 10).

**Note on performance:** G1 and G2 (earlier period grades) are deliberately excluded to avoid target leakage, so the app works as an early-warning system using only background and behavioural data. This makes the task much harder than predicting from earlier grades, so scores are modest on this small dataset (395 students).

## Project Structure

```
├── app.py            # Streamlit web app
├── train_model.py    # Model training script
├── data/             # Dataset (student-mat.csv)
├── models/           # Trained models (grade_model, risk_model)
├── notebooks/        # EDA notebooks
└── requirements.txt  # Dependencies
```

## How to Run

```bash
git clone https://github.com/omiii24/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Screenshot

![App screenshot](screenshot.png)

## Future Improvements

- Add feature-importance / SHAP explanations for each prediction
- Compare more models (e.g. XGBoost)
- Deploy on Streamlit Community Cloud

## Author

**Om Pawar**, B.Sc. IT, University of Mumbai
