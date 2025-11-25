
import os, random, time, json
import torch
import numpy as np
import pandas as pd
import gradio as gr
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define device for model operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "./agent_workspace/datasets"  # path for saving and loading dataset(s) (or the user's uploaded dataset) for preprocessing, training, hyperparamter tuning, deployment, and evaluation
MODEL_SAVE_PATH = "./agent_workspace/trained_models"  # path for saving the trained model

# Ensure the model save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Data preprocessing and feature engineering
def preprocess_data(data):
    # Normalize or standardize the features using MinMaxScaler from sklearn.preprocessing:
    scaler = MinMaxScaler()
    for column in data.columns:
        data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

    # Handle outliers by removing them using the Interquartile Range (IQR):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data > lower_bound) & (data < upper_bound)]

    # Verify there are no missing values as specified. If missing values are found, you can choose to either remove the entire row or fill the missing values with a suitable replacement (e.g., mean, median, or mode).
    data = data.dropna()

    return data

class TabularDataset(Dataset):
    def __init__(self, data, target_column):
        self.features = data.drop(target_column, axis=1).values.astype(np.float32)
        self.labels = data[target_column].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, train_loader, valid_loader, epochs=100):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for batch_features, batch_labels in valid_loader:
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_labels).item()
                pred = (outputs > 0.5).float()
                correct += (pred == batch_labels).sum().item()
                total += batch_labels.size(0)

            val_loss /= len(valid_loader)
            val_accuracy = correct / total

        print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return model

def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            test_loss += criterion(outputs, batch_labels).item()
            pred = (outputs > 0.5).float()
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)

        test_loss /= len(test_loader)
        test_accuracy = correct / total

    performance_scores = {
        'ACC': test_accuracy,
        'F1': f1_score(test_labels.cpu().numpy().flatten(), (outputs > 0.5).cpu().numpy().flatten())
    }

    return performance_scores

def prepare_model_for_deployment():
    # No specific steps needed for deployment in this case
    return model

def deploy_model():
    # Deploy the model using Gradio
    def predict(features):
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(features)
        return output.item()

    iface = gr.Interface(fn=predict, inputs="text", outputs="number")
    url_endpoint = iface.launch()
    return url_endpoint

# The main function to orchestrate the data loading, data preprocessing, feature engineering, model training, model preparation, model deployment, and model evaluation
def main():
    """
    Main function to execute the tabular classification pipeline.
    """

    # Step 1. Retrieve or load a dataset from hub (if available) or user's local storage (if given)
    dataset_path = os.path.join(DATASET_PATH, "banana_quality.csv")
    data = pd.read_csv(dataset_path)

    # Step 2. Create a train-valid-test split of the data by splitting the `dataset` into train_loader, valid_loader, and test_loader.
    # Here, the train_loader contains 70% of the `dataset`, the valid_loader contains 20% of the `dataset`, and the test_loader contains 10% of the `dataset`.
    X_train, X_val, y_train, y_val = train_test_split(data.drop('Quality', axis=1), data['Quality'], test_size=0.3, random_state=42)
    X_test, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    train_dataset = TabularDataset(X_train, 'Quality')
    valid_dataset = TabularDataset(X_val, 'Quality')
    test_dataset = TabularDataset(X_test, 'Quality')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Step 3. With the split dataset, run data preprocessing and feature engineering (if applicable) using the "preprocess_data" function you defined
    processed_data = preprocess_data(data)

    # Step 4. Define required model. You may retrieve model from available hub or library along with pretrained weights (if any).
    # If pretrained or predefined model is not available, please create the model according to the given user's requirements below using PyTorch and relevant libraries.
    class LogisticRegression(nn.Module):
        def __init__(self, input_size, output_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, output_size)

        def forward(self, x):
            out = self.linear(x)
            return torch.sigmoid(out)

    model = LogisticRegression(input_size=processed_data.shape[1], output_size=1).to(device)

    # Step 5. train the retrieved/loaded model using the defined "train_model" function
    # TODO: on top of the model training, please run hyperparamter optimization based on the suggested hyperparamters and their values before proceeding to the evaluation step to ensure model's optimality

    model = train_model(model, train_loader, valid_loader)

    # Step 6. evaluate the trained model using the defined "evaluate_model" function
    model_performance = evaluate_model(model, test_loader)

    # Step 7. compress and convert the trained model according to a given deployment platform using the defined "prepare_model_for_deployment" function
    deployable_model = prepare_model_for_deployment()

    # Step 8. deploy the model using the defined "deploy_model" function
    url_endpoint = deploy_model()

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'logistic_regression_model.pt'))

    return (
        processed_data,
        model,
        deployable_model,
        url_endpoint,
        model_performance,
    )

if __name__ == "__main__":
    processed_data, model, deployable_model, url_endpoint, model_performance = main()
    print("Model Performance on Test Set:", model_performance)
