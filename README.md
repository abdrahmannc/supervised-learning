# Supervised Learning

## Table of Contents
- [Description](#description)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Models & Metrics](#models--metrics)

---

## Description
This repository is designed to learn and showcase supervised learning workflows, including data preparation, model training, evaluation, and overfitting detection. It covers both **regression** (continuous predictions) and **classification** (categorical predictions).

---

## Folder Structure

- `regression/` - Regression experiments (Linear Regression, Decision Tree, Random Forest)  
- `classification/` - Classification experiments (Logistic Regression, SVM, Random Forest)  
- `notebooks/` - Shared notebooks for exploration and tutorials  
- `data/` - Folder containing all datasets used in the projects  
- `README.md` - This file  

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abdrahmannc/supervised-learning.git
2. Navigate to the repository:
    ```bash
    cd supervised-learning
3. Install required packages:
    ```bash 
     pip install -r requirements.txt
## Datasets

All datasets used in this repository are stored in the `data/` folder.  
This includes both regression and classification datasets used for training, validation, and testing the models in the notebooks and scripts.

### Regression

**HM SENSE Sensor Data (sensor09)** – One month of environmental measurements (temperature, humidity, CO₂, light, motion) from sensor09 at Hochschule München, in CSV format (01.12.2025 – 31.12.2025).  


This dataset is used in this project for regression tasks, such as predicting room temperature from sensor readings.


  File: `data/sensor09__20251201_20251231.csv`
  
  https://hm-sense-open-data-api.kube.cs.hm.edu/api/roomclimate/measurements/sensor09?start=1764547200&end=1767225599&format=csv

  Note:
  This dataset from Hochschule München is an example distribution for API demonstration.
  The measurements are not real operational data and are provided for educational and testing purposes only.

### Classification
- **** – 


**Note:** Users should download the datasets and place them in the `data/` folder as instructed in each project.


## Models & Metrics

### Regression
- **Models**
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- **Metrics**
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R-squared 

### Classification
- **Models**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
- **Metrics**
  - 0–1 Loss / Accuracy
  - Cross-Entropy Loss
  - Precision, Recall, F1-score 
