## Overview
This project analyzes customer churn using:
- Exploratory Data Analysis (EDA)
- Business Value Modeling (Expected Value)
- Machine Learning (Naive Bayes)

---

## Project Structure

### 1. analysis.py
Exploratory data analysis:
- churn probability analysis
- conditional probabilities
- lift analysis
- feature impact evaluation

### 2. value_model.py
Business impact modeling:
- Expected Value (EV) per customer
- segmentation (risk/value)
- campaign optimization under budget
- scenario/sensitivity analysis

### 3. ml_model.py
Predictive modeling:
- custom Naive Bayes implementation
- sklearn CategoricalNB comparison
- evaluation metrics (Accuracy, F1, Recall)

---

## Pipeline

EDA → Business modeling → ML prediction

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- NumPy

---

## How to run

```bash
python analysis.py
python value_model.py
python ml_model.py