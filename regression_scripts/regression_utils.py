
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_style('whitegrid')

# -----------------------------
# 1️⃣ Load CSV
# -----------------------------
def load_data(file_path, sep=","):
    """
    Load a CSV file into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at {file_path}.")
    data = pd.read_csv(file_path, sep=sep)
    print(f"✅ Loaded data with shape {data.shape}")
    return data

# -----------------------------
# 2 Explore data
# -----------------------------
def explore_data(data,target_col):
    target=data[target_col]
    print("Description")
    print(data.describe())

    #the heatmap
    X_numeric = data.select_dtypes(include="number").corr()

    sns.heatmap(
        X_numeric,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        square=True
    )

    plt.title("Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.show()

    #numeric cols
    numeric_cols = data.select_dtypes(include="number").columns.drop(target_col)
    for col in numeric_cols.columns:
        plt.figure()
        sns.regplot(
            x=data[col],
            y=target,
            scatter_kws={
                "alpha": 0.6,
                "color": "blue",
                "label": "Observation"
            },
            line_kws={
                "alpha": 0.8,
                "color": "orange",
                "label": f"{col} effect"
            }
        )

        plt.title(f"{col} vs {target.name}")
        plt.xlabel(col)
        plt.ylabel(target.name)
        plt.legend()
        plt.show()


