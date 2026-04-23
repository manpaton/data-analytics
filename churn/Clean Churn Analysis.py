import pandas as pd
import matplotlib.pyplot as plt


# ---------------- LOAD DATA ----------------

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------- CORE PROBABILITY ----------------

def churn_rate(df: pd.DataFrame) -> float:
    return (df['Churn'] == 'Yes').mean()


def safe_churn_rate(df: pd.DataFrame) -> float:
    counts = df['Churn'].value_counts(normalize=True)
    return counts.get('Yes', 0)


# ---------------- ANALYSIS HELPERS ----------------

def group_churn(df: pd.DataFrame, feature: str) -> pd.Series:
    return df.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').mean())


def feature_diff_strength(df: pd.DataFrame, feature: str, base_rate: float) -> float:
    max_diff = 0
    for cat in df[feature].dropna().unique():
        subset = df[df[feature] == cat]
        diff = abs(safe_churn_rate(subset) - base_rate)
        max_diff = max(max_diff, diff)
    return max_diff


# ---------------- MAIN ----------------

def main():
    df = load_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    base_p_churn = churn_rate(df)
    print("Base churn probability:", base_p_churn)

    # 1.2 conditional probabilities
    mtm = df[df['Contract'] == 'Month-to-month']
    print("MTM churn:", safe_churn_rate(mtm))

    tenure_low = df[df['tenure'] < 12]
    print("Low tenure churn:", safe_churn_rate(tenure_low))

    fiber = df[df['InternetService'] == 'Fiber optic']
    print("Fiber churn:", safe_churn_rate(fiber))

    electronic_pay = df[df['PaymentMethod'] == 'Electronic check']
    print("Electronic payment churn:", safe_churn_rate(electronic_pay))

    # 1.3 lift
    mtm_fiber = df[(df['Contract'] == 'Month-to-month') & (df['InternetService'] == 'Fiber optic')]
    lift = safe_churn_rate(mtm_fiber) / base_p_churn
    print("Lift (MTM + Fiber):", lift)

    # 1.4 comparisons
    contracts = {
        "One year": df[df['Contract'] == 'One year'],
        "Two year": df[df['Contract'] == 'Two year'],
        "Month-to-month": df[df['Contract'] == 'Month-to-month']
    }

    print("Min churn among contracts:", min(safe_churn_rate(v) for v in contracts.values()))

    sc = df[df['SeniorCitizen'] == 1]
    nsc = df[df['SeniorCitizen'] == 0]

    print("Senior vs non-senior:",
          "higher" if safe_churn_rate(sc) > safe_churn_rate(nsc) else "lower")

    paper_yes = df[df['PaperlessBilling'] == 'Yes']
    paper_no = df[df['PaperlessBilling'] == 'No']

    print("Paperless diff:", safe_churn_rate(paper_yes) - safe_churn_rate(paper_no))

    # 1.6 feature strength
    categorical_features = [
        'Contract', 'PaymentMethod', 'InternetService', 'PaperlessBilling',
        'SeniorCitizen', 'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV'
    ]

    feature_strength = {}

    for feature in categorical_features:
        feature_strength[feature] = feature_diff_strength(df, feature, base_p_churn)

    top5 = sorted(feature_strength.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_names = [x[0] for x in top5]

    # visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for ax, feature in zip(axes, top5_names):
        group_churn(df, feature).plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(feature)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()