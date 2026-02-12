import os
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

# Define GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define device for model operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Define data path
DATASET_PATH = "./datasets/banana_quality.csv"
MODEL_PATH = "./trained_models/"

# Load dataset
def load_dataset(path):
    df = pd.read_csv(path)
    return df

# Preprocess data
def preprocess_data(df):
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

# Train model
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Define model
    model = LogisticRegression()

    # Define hyperparameters for grid search
    param_grid = {'C': [0.1, 1, 10, 100], 'l1_ratio': [0.0, 0.5, 1.0]}

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}, F1: {f1}')

    # Save model
    joblib.dump(best_model, os.path.join(MODEL_PATH, 'best_model.pkl'))

    return best_model

# Main function
def main():
    # Load dataset
    df = load_dataset(DATASET_PATH)

    # Preprocess data
    df = preprocess_data(df)

    # Train model
    model = train_model(df)

if __name__ == "__main__":
    main()