import pandas as pd


# ---------------- CONFIG ----------------

CLV = 2000
SUCCESS_RATE = 0.4
COST = 300
BUDGET = 50000


# ---------------- LOAD ----------------

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------- FEATURE ENGINEERING ----------------

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # simplified churn probability assumption (placeholder model)
    df['P_churn'] = df['Churn'].map({'Yes': 0.7, 'No': 0.3})

    # expected value baseline
    df['EV'] = df['P_churn'] * CLV * SUCCESS_RATE - COST

    # segmentation
    df['Value'] = df['MonthlyCharges'].apply(lambda x: 'High' if x > 70 else 'Low')
    df['Risk'] = df['P_churn'].apply(lambda x: 'High' if x > 0.6 else 'Low')

    return df


# ---------------- ANALYSIS ----------------

def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['Value', 'Risk']).agg(
        num_customers=('customerID', 'count'),
        expected_loss=('EV', lambda x: (-x[x < 0]).sum()),
        expected_gain=('EV', lambda x: x[x > 0].sum())
    ).reset_index()


def select_top_customers(df: pd.DataFrame, budget: int, cost_per_customer: int, ev_col: str):
    max_customers = budget // cost_per_customer
    return df.sort_values(ev_col, ascending=False).head(max_customers)


def campaign_result(df: pd.DataFrame, selected: pd.DataFrame, ev_col: str):
    total_profit = selected[ev_col].sum()

    high_risk_total = (df['P_churn'] > 0.6).sum()
    high_risk_selected = (selected['P_churn'] > 0.6).sum()

    coverage = (high_risk_selected / high_risk_total) * 100 if high_risk_total > 0 else 0

    return total_profit, coverage


# ---------------- SCENARIOS ----------------

def run_scenarios(df: pd.DataFrame):
    df = df.copy()

    df['EV_s1'] = df['P_churn'] * CLV * 0.2 - COST
    df['EV_s2'] = df['P_churn'] * CLV * SUCCESS_RATE - 500
    df['EV_s3'] = df['P_churn'] * 1500 * SUCCESS_RATE - COST

    scenarios = {
        'Scenario 1': ('EV_s1', 50000, COST),
        'Scenario 2': ('EV_s2', 50000, 500),
        'Scenario 3': ('EV_s3', 50000, COST)
    }

    print("\n--- Sensitivity Analysis ---\n")

    for name, (ev_col, budget, cost_per_customer) in scenarios.items():
        selected = select_top_customers(df, budget, cost_per_customer, ev_col)

        profit, coverage = campaign_result(df, selected, ev_col)

        recommendation = "RUN" if profit > 0 else "DO NOT RUN"

        print(f"{name}")
        print(f"Customers targeted: {len(selected)}")
        print(f"Expected profit: ${profit:.2f}")
        print(f"High-risk coverage: {coverage:.2f}%")
        print(f"Recommendation: {recommendation}\n")


# ---------------- MAIN ----------------

def main():
    df = load_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df = prepare_data(df)

    print(df[['customerID', 'Churn', 'P_churn', 'EV']].head())

    matrix = build_matrix(df)
    print("\nSegment Matrix:\n")
    print(matrix)

    selected = select_top_customers(df, BUDGET, COST, 'EV')

    profit, coverage = campaign_result(df, selected, 'EV')

    print("\n--- Base Strategy ---")
    print(f"Customers targeted: {len(selected)}")
    print(f"Expected profit: ${profit:.2f}")
    print(f"High-risk coverage: {coverage:.2f}%")

    run_scenarios(df)


if __name__ == "__main__":
    main()