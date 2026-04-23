import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


# ---------------- LOAD DATA ----------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.dropna(subset=['metascore'], inplace=True)
    return df


# ---------------- CLEANING ----------------

def detect_bad_data(df: pd.DataFrame):
    bad_budget = df[df["budget"] <= 0]
    bad_time = df[df["runtime"] <= 60]
    bad_revenue = df[df["revenue"] <= 0]
    bad_rating = df[(df["imdb_rating"] <= 0) | (df["metascore"] <= 0)]

    print("Bad budget:", len(bad_budget))
    print("Bad runtime:", len(bad_time))
    print("Bad revenue:", len(bad_revenue))
    print("Bad ratings:", len(bad_rating))


# ---------------- STATISTICS ----------------

def numerical_summary(df: pd.DataFrame):
    numeric_cols = [
        "budget", "revenue", "runtime", "imdb_rating",
        "metascore", "votes", "oscar_nominations", "oscar_wins"
    ]

    summary = []

    for col in numeric_cols:
        s = df[col].dropna()
        summary.append({
            "column": col,
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "iqr": s.quantile(0.75) - s.quantile(0.25)
        })

    return pd.DataFrame(summary)


# ---------------- VISUALIZATION ----------------

def plot_distributions(df: pd.DataFrame):
    numeric_cols = [
        "budget", "revenue", "runtime", "imdb_rating",
        "metascore", "votes"
    ]

    df[numeric_cols].hist(bins=30, figsize=(12, 8))
    plt.suptitle("Distributions")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


# ---------------- OUTLIERS ----------------

def get_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df[(df[col] < lower) | (df[col] > upper)]


# ---------------- MAIN ANALYSIS ----------------

def main():
    df = load_data("movies.csv")

    print("Dataset size:", len(df))

    detect_bad_data(df)

    print("\nNumerical Summary:")
    print(numerical_summary(df))

    print("\nTop rated movies without Oscar:")
    top = df[df["imdb_rating"] > 8.5]
    print(top[top["oscar_nominations"] == 0][["title", "imdb_rating"]].head())

    # Outliers
    outliers = get_outliers(df, "budget")
    print("Budget outliers:", len(outliers))

    # Stats interpretation
    numeric_cols = ["budget", "revenue", "runtime", "imdb_rating"]

    print("\nCoefficient of Variation:")
    for col in numeric_cols:
        cv = df[col].std() / df[col].mean()
        print(col, round(cv, 2))

    # Visualization
    plot_distributions(df)

    # Normality check
    stats.probplot(df["imdb_rating"], dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()


if __name__ == "__main__":
    main()
