# 🎓 Student Performance Prediction System

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
| Grade model (regression) | MAE | XX |
| Grade model (regression) | R² | XX |
| Risk model (classification) | Accuracy | XX |
| Risk model (classification) | F1-score | XX |

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
