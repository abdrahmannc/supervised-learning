
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split dataset with proper stratification.
    
    Parameters:
    - train_size: proportion for training (default 0.7)
    - val_size: proportion for validation (default 0.15)
    - test_size: proportion for testing (default 0.15)
    """
    # Verify splits sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Splits must sum to 1"
    
    # Always stratify for classification (few unique values)
    is_classification = len(np.unique(y)) < 50
    
    # First split: train vs (val+test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, y,
        train_size=train_size,
        stratify=y if is_classification else None,
        random_state=random_state
    )
    
    # Second split: val vs test from temp
    # Calculate what proportion of temp should be test
    test_of_temp = test_size / (val_size + test_size)
    
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=test_of_temp,
        stratify=Y_temp if is_classification else None,
        random_state=random_state
    )
    
    # Verify distributions
    if is_classification:
        print("\n📊 Class distribution check:")
        original_dist = pd.Series(y).value_counts(normalize=True).sort_index()
        train_dist = pd.Series(Y_train).value_counts(normalize=True).sort_index()
        val_dist = pd.Series(Y_val).value_counts(normalize=True).sort_index()
        test_dist = pd.Series(Y_test).value_counts(normalize=True).sort_index()
        
        print(f"Original:  {original_dist.values}")
        print(f"Train:     {train_dist.values}")
        print(f"Val:       {val_dist.values}")
        print(f"Test:      {test_dist.values}")
    
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



# -----------------------------
# 7 confusion matrix
# -----------------------------
def plot_confusion_matrix(model, X, y, title="Confusion Matrix"):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=model.classes_)

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_,
                yticklabels=model.classes_)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()