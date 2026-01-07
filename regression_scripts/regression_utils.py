
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
