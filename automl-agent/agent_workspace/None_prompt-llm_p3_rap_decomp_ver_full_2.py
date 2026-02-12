import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

# Define data path
DATASET_PATH = "./datasets/banana_quality.csv"
MODEL_PATH = "./trained_models/"

# Load dataset
def load_dataset(path):
    df = pd.read_csv(path)
    return df

# Preprocess data
def preprocess_data(df):
    # Check and convert non-numeric columns to numeric if possible
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except ValueError:
            print(f"Column {col} could not be converted to numeric.")
            df[col] = df[col].fillna(df[col].mode().iloc[0])  # Fill with mode

    # Remove outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Handle missing values
    df_out.fillna(df_out.mean(), inplace=True)

    # Normalize/Standardize
    scaler = StandardScaler()
    df_out = pd.DataFrame(scaler.fit_transform(df_out), columns=df_out.columns)

    # Feature augmentation
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    df_out = pd.DataFrame(poly.fit_transform(df_out), columns=poly.get_feature_names(df_out.columns))

    # PCA
    pca = PCA(n_components=0.95)
    df_out = pd.DataFrame(pca.fit_transform(df_out), columns=[f'PC{i}' for i in range(1, pca.n_components_+1)])

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=10)
    df_out = pd.DataFrame(selector.fit_transform(df_out, df['target']), columns=selector.get_support(indices=True))

    return df_out

# Rest of the code remains the same...