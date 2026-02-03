
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.base import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
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
    numeric_cols = data.select_dtypes(include="number")
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



# -----------------------------
# 3 Process data
# -----------------------------
def data_process(data, target_col):
    """
    Preprocess a dataset for machine learning.

    Steps:
    1. Separates the target column from features.
    2. Selects numeric features and drops the target column.
    3. One-hot encodes categorical (object) columns with drop_first=True.
    4. Concatenates numeric and encoded categorical features.

    Returns:
        X (pd.DataFrame): Preprocessed feature matrix.
        target (pd.Series): Target column.
    """
    target = data[target_col]

    X = data.drop(columns=[target_col])
    y=data[target_col]
    #X = pd.concat([X_numeric, y], axis=1)

    return X, y



# -----------------------------
# 4 spliting data
# -----------------------------
"""
Split dataset into train, validation, and test sets.
Applies stratified splitting if the target is categorical.
"""

def split_data(X, y, trainSize=0.7):
    # Check if target is categorical
    stratify_val = y if y.dtype == "object" or str(y.dtype) == "category" else None
    
    # Split train vs temp
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, y, train_size=trainSize, stratify=stratify_val, random_state=42
    )
    
    # Split temp → val + test evenly
    stratify_temp = Y_temp if stratify_val is not None else None
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, stratify=stratify_temp, random_state=42
    )
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# -----------------------------
# 5 Train regression tree
# -----------------------------
def train_regression_tree(X_train, Y_train, max_depth=None, min_samples_split=2):

    
    """
    Train a DecisionTreeRegressor.
    """
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train, Y_train)
    print(f"✅ Regression Tree trained with max_depth={max_depth}")
    return model
