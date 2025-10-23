
# The following code is for "tabular classification" task.
import os, random, time, json

# define GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import numpy as np
import pandas as pd
import gradio as gr

# TODO: import other required library here, including libraries for datasets and (pre-trained) models like HuggingFace and Kaggle APIs. If the required module is not found, you can directly install it by running `pip install your_module`.
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define device for model operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "_experiments/datasets"  # path for saving and loading dataset(s) (or the user's uploaded dataset) for preprocessing, training, hyperparamter tuning, deployment, and evaluation

# Data preprocessing and feature engineering
def preprocess_data(dataset):
    # Normalize the features using MinMaxScaler from sklearn
    scaler = MinMaxScaler()
    dataset[['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']] = scaler.fit_transform(dataset[['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']])
    return dataset

def train_model(dataset):
    # Split the dataset into training, validation, and testing sets
    X = dataset.drop('Quality', axis=1)
    y = dataset['Quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Create a Logistic Regression model using sklearn
    model = LogisticRegression()

    # Fit the model on the training data and evaluate it on the validation data
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy: {val_accuracy}")

    # Optimize the hyperparameters of the model using GridSearchCV
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the testing data
    y_pred_test = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Testing Accuracy: {test_accuracy}")

    return best_model

def evaluate_model(model, dataset):
    # Split the dataset into training, validation, and testing sets
    X = dataset.drop('Quality', axis=1)
    y = dataset['Quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Evaluate the best model on the testing data
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Testing Accuracy: {test_accuracy}")

    performance_scores = {
        'ACC': test_accuracy,
        'F1': f1_score(y_test, y_pred_test, average='weighted')
    }

    return performance_scores

def prepare_model_for_deployment(model):
    # Save the best model to a PyTorch-compatible format using joblib
    joblib.dump(model, './agent_workspace/trained_models/best_model.pkl')

    return model

def deploy_model():
    # Deploy the model using the Gradio Python library
    def predict(size, weight, sweetness, softness, harvest_time, ripeness, acidity):
        model = joblib.load('./agent_workspace/trained_models/best_model.pkl')
        input_data = [[size, weight, sweetness, softness, harvest_time, ripeness, acidity]]
        prediction = model.predict(input_data)
        return prediction[0]

    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.inputs.Number(label="Size"),
            gr.inputs.Number(label="Weight"),
            gr.inputs.Number(label="Sweetness"),
            gr.inputs.Number(label="Softness"),
            gr.inputs.Number(label="Harvest Time"),
            gr.inputs.Number(label="Ripeness"),
            gr.inputs.Number(label="Acidity")
        ],
        outputs="text",
        title="Banana Quality Prediction",
        description="Enter the features of a banana to predict its quality."
    )

    url_endpoint = iface.launch(share=True)
    return url_endpoint

# The main function to orchestrate the data loading, data preprocessing, feature engineering, model training, model preparation, model deployment, and model evaluation
def main():
    """
    Main function to execute the tabular classification pipeline.
    """

    # Retrieve the 'BananaQualityDataset' from the specified source
    dataset = pd.read_csv('./agent_workspace/datasets/banana_quality.csv')

    # Preprocess the dataset
    processed_data = preprocess_data(dataset)

    # Train the model
    model = train_model(processed_data)

    # Evaluate the model
    model_performance = evaluate_model(model, processed_data)

    # Prepare the model for deployment
    deployable_model = prepare_model_for_deployment(model)

    # Deploy the model
    url_endpoint = deploy_model()

    return (
        processed_data,
        model,
        deployable_model,
        url_endpoint,
        model_performance
    )

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance = main()
    print("Model Performance on Test Set:", model_performance)
