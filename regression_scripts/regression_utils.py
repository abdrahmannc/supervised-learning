
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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
def explore_data(data, target_col):
    print("Description")
    print(data.describe())

    # Target distribution
    plt.figure()
    sns.countplot(x=data[target_col])
    plt.title("Target Distribution")
    plt.show()

    # Correlation heatmap (numeric only)
    X_numeric = data.select_dtypes(include="number").corr()
    sns.heatmap(X_numeric, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.show()

    # Boxplots per class
    numeric_cols = data.select_dtypes(include="number").columns
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=data[target_col], y=data[col])
        plt.title(f"{col} distribution by {target_col}")
        plt.show()



# -----------------------------
# 3 Process data
# -----------------------------
def data_process(data, target_col, feature_cols=None):
    """
    Preprocess a dataset for machine learning.

    Preprocess dataset for DecisionTreeClassifier with a categorical target.
    Returns:
        X (pd.DataFrame): Preprocessed feature matrix.
    """
    target = data[target_col]

    if feature_cols is not None:
        X=data[feature_cols].copy()

    else:
        X=data.drop(columns=[target_col])
        

    return X, target



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
# 5 train model
# -----------------------------

def train_tree_classifier(X_train, Y_train, max_depth=3):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, Y_train)
    return model



# -----------------------------
# 6 evaluate model
# -----------------------------

def evaluate_tree_classifier_model(model, X, y, name="Dataset"):
    y_pred = model.predict(X)

    print(f"\n{name} Evaluation")
    print("-" * 30)
    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    return y_pred


