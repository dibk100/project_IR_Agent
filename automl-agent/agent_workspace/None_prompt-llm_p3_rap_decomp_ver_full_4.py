
import os, random, time, json

# Define GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define device for model operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "_experiments/datasets"  # path for saving and loading dataset(s) (or the user's uploaded dataset) for preprocessing, training, hyperparamter tuning, deployment, and evaluation

# Custom Dataset class
class TabularDataset(Dataset):
    def __init__(self, X, y, tokenizer=None, max_length=512):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]
        label = self.y[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Data preprocessing and feature engineering
def preprocess_data(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Split dataset into features and target
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Preprocessing steps
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TabularDataset(X=X_train, y=y_train, tokenizer=tokenizer)
    val_dataset = TabularDataset(X=X_val, y=y_val, tokenizer=tokenizer)
    test_dataset = TabularDataset(X=X_test, y=y_test, tokenizer=tokenizer)

    return train_dataset, val_dataset, test_dataset, tokenizer

def train_model(train_dataset, val_dataset, tokenizer):
    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(train_dataset.y)))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train model
    trainer.train()

    return model

def evaluate_model(model, test_dataset):
    # Evaluate model
    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
    )
    results = trainer.evaluate()

    # Extract performance scores
    performance_scores = {
        'ACC': results['eval_accuracy'],
        'F1': results['eval_f1']
    }

    return performance_scores

def prepare_model_for_deployment(model):
    # Save model
    model.save_pretrained('./agent_workspace/trained_models')

    return model

def deploy_model():
    # Deploy model using Gradio
    def predict(text):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.item()

    demo = gr.Interface(fn=predict, inputs="text", outputs="label")
    demo.launch()
    return demo.url

# The main function to orchestrate the data loading, data preprocessing, feature engineering, model training, model preparation, model deployment, and model evaluation
def main():
    """
    Main function to execute the tabular classification pipeline.
    """

    # Step 1. Retrieve or load a dataset from hub (if available) or user's local storage (if given)
    dataset_path = "path_to_your_dataset.csv"  # Replace with actual dataset path
    train_dataset, val_dataset, test_dataset, tokenizer = preprocess_data(dataset_path)

    # Step 2. Train the retrieved/loaded model using the defined "train_model" function
    model = train_model(train_dataset, val_dataset, tokenizer)

    # Step 3. Evaluate the trained model using the defined "evaluate_model" function
    model_performance = evaluate_model(model, test_dataset)

    # Step 4. Compress and convert the trained model according to a given deployment platform using the defined "prepare_model_for_deployment" function
    deployable_model = prepare_model_for_deployment(model)

    # Step 5. Deploy the model using the defined "deploy_model" function
    url_endpoint = deploy_model()

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
        model,
        deployable_model,
        url_endpoint,
        model_performance,
    )

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, tokenizer, model, deployable_model, url_endpoint, model_performance = main()
    print("Model Performance on Test Set:", model_performance)
