import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder


# ---------------- DATA PREP ----------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    df['tenure_group'] = pd.qcut(df['tenure'], 4, labels=['Short', 'Medium', 'Long', 'VeryLong'])
    df['MonthlyCharges_group'] = pd.qcut(df['MonthlyCharges'], 3, labels=['Low', 'Medium', 'High'])
    df['TotalCharges_group'] = pd.qcut(df['TotalCharges'], 3, labels=['Low', 'Medium', 'High'])

    return df


# ---------------- SPLIT ----------------

def split_data(df):
    return train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['Churn']
    )


# ---------------- CUSTOM NAIVE BAYES ----------------

def compute_priors(train_df):
    return train_df['Churn'].value_counts(normalize=True)


def compute_feature_probs(train_df, categorical_features):
    feature_probs = {}

    for feature in categorical_features:
        yes_probs = train_df[train_df['Churn'] == 'Yes'][feature].value_counts(normalize=True)
        no_probs = train_df[train_df['Churn'] == 'No'][feature].value_counts(normalize=True)

        feature_probs[feature] = {'Yes': yes_probs, 'No': no_probs}

    return feature_probs


def predict_row(row, priors, feature_probs, categorical_features):
    yes_prob = priors['Yes']
    no_prob = priors['No']

    for feature in categorical_features:
        value = row[feature]

        yes_prob *= feature_probs[feature]['Yes'].get(value, 1e-6)
        no_prob *= feature_probs[feature]['No'].get(value, 1e-6)

    pred = 'Yes' if yes_prob > no_prob else 'No'
    prob = yes_prob / (yes_prob + no_prob)

    return pred, prob


# ---------------- SKLEARN MODEL ----------------

def sklearn_model(train_df, test_df, categorical_features):
    X_train = train_df[categorical_features].copy()
    X_test = test_df[categorical_features].copy()

    encoders = {}

    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    model = CategoricalNB()
    model.fit(X_train, train_df['Churn'])

    return model, X_test


# ---------------- EVALUATION ----------------

def evaluate(y_true, y_pred, title="Model"):
    print(f"\n--- {title} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label='Yes'))
    print("Recall:", recall_score(y_true, y_pred, pos_label='Yes'))
    print("F1-score:", f1_score(y_true, y_pred, pos_label='Yes'))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


# ---------------- MAIN ----------------

def main():
    df = load_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    train_df, test_df = split_data(df)

    categorical_features = [
        'Contract', 'PaymentMethod', 'InternetService', 'PaperlessBilling', 'SeniorCitizen',
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'tenure_group', 'MonthlyCharges_group', 'TotalCharges_group'
    ]

    # -------- custom Naive Bayes --------
    priors = compute_priors(train_df)
    feature_probs = compute_feature_probs(train_df, categorical_features)

    predictions = []
    probs = []

    for _, row in test_df.iterrows():
        pred, prob = predict_row(row, priors, feature_probs, categorical_features)
        predictions.append(pred)
        probs.append(prob)

    test_df['predicted'] = predictions
    test_df['prob_churn'] = probs

    evaluate(test_df['Churn'], test_df['predicted'], "Custom Naive Bayes")

    # -------- sklearn Naive Bayes --------
    model, X_test = sklearn_model(train_df, test_df, categorical_features)

    y_pred_skl = model.predict(X_test)

    evaluate(test_df['Churn'], y_pred_skl, "Sklearn CategoricalNB")


if __name__ == "__main__":
    main()
