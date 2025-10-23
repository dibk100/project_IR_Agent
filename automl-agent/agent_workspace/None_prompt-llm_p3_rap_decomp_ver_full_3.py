
import os
import random
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
import gradio as gr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Define GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define device for model operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path for saving and loading dataset(s) (or the user's uploaded dataset) for preprocessing, training, hyperparameter tuning, deployment, and evaluation
DATASET_PATH = "_experiments/datasets"
MODEL_PATH = "_experiments/trained_models"

# Ensure directories exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Data preprocessing and feature engineering
def preprocess_data(dataset_path):
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Separate features and target
    X = data.drop('quality', axis=1).values
    y = data['quality'].values
    
    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle outliers by calculating the interquartile range (IQR)
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    mask = ~((X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))).any(axis=1)
    X_scaled = X_scaled[mask]
    y = y[mask]
    
    # Replace missing values with the median
    X_scaled = np.nan_to_num(X_scaled, nan=np.median(X_scaled))
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_scaled).float()
    y_tensor = torch.from_numpy(y).long()
    
    return X_tensor, y_tensor

def train_model(X_train, y_train, X_val, y_val):
    # Initialize the Logistic Regression model
    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    
    # Define hyperparameter grid for grid search
    param_grid = {'C': [0.1, 0.5, 1.0, 5.0, 10.0]}
    
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate the model on the validation set
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='binary')
    recall = recall_score(y_val, y_pred, average='binary')
    f1 = f1_score(y_val, y_pred, average='binary')
    auc_roc = roc_auc_score(y_val, best_model.decision_function(X_val), average='binary')
    
    print(f'Validation Accuracy: {accuracy}')
    print(f'Validation Precision: {precision}')
    print(f'Validation Recall: {recall}')
    print(f'Validation F1 Score: {f1}')
    print(f'Validation AUC-ROC: {auc_roc}')
    
    return best_model

def save_model(model, model_path):
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

# The main function to orchestrate the data loading, data preprocessing, feature engineering, model training, model preparation, model deployment, and model evaluation
def main():
    """
    Main function to execute the tabular classification pipeline.
    """
    
    # Step 1. Retrieve or load a dataset from user's local storage (if given)
    dataset_path = os.path.join(DATASET_PATH, 'banana_quality.csv')
    
    # Step 2. Preprocess the data
    X_tensor, y_tensor = preprocess_data(dataset_path)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Step 3. Train the model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Step 4. Save the trained model
    model_path = os.path.join(MODEL_PATH, 'logistic_regression.pth')
    save_model(model, model_path)
    
    return model_path

# Function to predict quality based on input features
def predict_quality(features):
    # Load the pre-trained model
    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'logistic_regression.pth')))
    model.eval()
    
    # Preprocess the input features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])
    features_tensor = torch.from_numpy(features_scaled).float()
    
    # Make prediction
    with torch.no_grad():
        prediction = model.predict(features_tensor)
    
    return prediction[0]

# Gradio interface for interactive testing
def gradio_interface():
    inputs = [
        gr.inputs.Number(label="Feature 1"),
        gr.inputs.Number(label="Feature 2"),
        gr.inputs.Number(label="Feature 3"),
        gr.inputs.Number(label="Feature 4")
    ]
    outputs = gr.outputs.Label(label="Predicted Quality")
    
    iface = gr.Interface(fn=predict_quality, inputs=inputs, outputs=outputs, title="Banana Quality Prediction")
    iface.launch()

if __name__ == "__main__":
    model_path = main()
    print(f"Trained model saved at {model_path}")
    
    # Launch Gradio interface for interactive testing
    gradio_interface()
