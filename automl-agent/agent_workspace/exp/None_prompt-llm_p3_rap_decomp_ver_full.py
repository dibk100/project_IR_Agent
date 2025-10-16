
import os, random, time, json

# define GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
import gradio as gr
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set the path for saving and loading dataset(s)
DATASET_PATH = "./agent_workspace/datasets"

# Set the path for saving and loading trained models
MODEL_PATH = "./agent_workspace/trained_models"

# Data preprocessing
def preprocess_data():
    # Load the dataset
    dataset = pd.read_csv(f"{DATASET_PATH}/banana_quality.csv")

    # Normalize the features
    scaler = StandardScaler()
    dataset.drop("quality", axis=1, inplace=True)  # Remove the target column
    X = dataset.values
    scaler.fit(X)
    X = scaler.transform(X)
    dataset = pd.DataFrame(X, columns=dataset.columns)
    dataset["quality"] = dataset["quality"].astype(int)  # Convert quality column to integer

    # Handle outliers and missing values
    dataset = dataset.dropna()

    # Generate polynomial features and interaction terms
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X = poly.fit_transform(dataset)

    # Apply Principal Component Analysis (PCA) for dimensionality reduction if necessary
    pca = PCA(n_components=0.95)
    X = pca.fit_transform(X)

    return dataset, X

# Data preparation
def prepare_data(dataset, X):
    y = dataset["quality"]
    dataset.drop(["quality"], axis=1, inplace=True)
    return dataset, X, y

# Model Development
def create_logistic_regression_model():
    input_size = X.shape[1]
    output_size = 1
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(64, output_size)
    ).to(device)
    return model

# Model Training
def train_model(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    train_loss = []
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
    return train_loss

# Model Evaluation
def evaluate_model(model, test_loader, criterion):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1

# Model Saving
def save_model(model, filename):
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, filename))

# Main Function
def main():
    # Load or preprocess the dataset
    dataset, X = preprocess_data()
    dataset, X, y = prepare_data(dataset, X)

    # Split the data into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(X))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(X) - train_size])
    train_X, val_X = torch.utils.data.random_split(X, [train_size, len(X) - train_size])
    train_y, val_y = torch.utils.data.random_split(y, [train_size, len(X) - train_size])

    # Create the model
    model = create_logistic_regression_model()

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    # Train the model
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_loss = train_model(model, train_loader, optimizer, criterion, epochs=10)

    # Evaluate the model
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion)

    # Save the trained model
    save_model(model, "banana_quality_logistic_regression.pth")

    return val_accuracy, val_f1

# Web Application Demo using the Gradio library
def demo():
    input_shape = (1, X.shape[1])
    model = torch.load(os.path.join(MODEL_PATH, "banana_quality_logistic_regression.pth"))
    model.eval()

    def predict(input_data):
        input_data = torch.from_numpy(input_data.reshape(input_shape)).to(device)
        with torch.no_grad():
            outputs = model(input_data)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

    interface = gr.Interface(fn=predict, inputs="auto", outputs="number",
                              title="Banana Quality Classification",
                              description="Enter the features of a banana to predict its quality.")
    interface.launch()

if __name__ == "__main__":
    accuracy, f1 = main()
    print(f"Validation Accuracy: {accuracy}")
    print(f"Validation F1 Score: {f1}")
