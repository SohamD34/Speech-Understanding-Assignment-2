import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from sklearn.model_selection import GridSearchCV
from utils import *
warnings.filterwarnings("ignore")

os.chdir('../../')

def load_data(data_path):
    """
    Load the dataset from the specified path.
    """
    data = pd.read_csv(data_path)
    return data


def preprocess_data(data):

    """
    Preprocess the dataset by splitting it into features and labels.
    """
    X = data.drop(columns=['label'])
    y = data['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    encoding = dict(zip(encoder.classes_, range(len(encoder.classes_))))

    return X_scaled, y_encoded, encoding



def train_model(model, X_train, y_train):
    """
    Train the specified model on the training data.
    """
    model.fit(X_train, y_train)
    training_acc = model.score(X_train, y_train)

    return model, training_acc



def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data and print the classification report.
    """
    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred)
    test_acc = model.score(X_test, y_test)

    return clf_report, test_acc



if __name__ == '__main__':

    # Data import and preprocessing

    dataset = load_data('data/audio_dataset/mel_features.csv')
    X, y, encoding = preprocess_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


    # Defining hyperparameter grids for each model

    param_grids = {
        'SVC': {'kernel': ['linear', 'rbf','sigmoid','poly'], 'C': [0.1, 1, 10]},
        'RFC': {'n_estimators': [50, 75, 100, 125, 150, 175, 200], 'max_depth': [None, 5, 10, 15, 20]},
        'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    }

    models = {
        'SVC': SVC(),
        'RFC': RFC(),
        'KNN': KNN()
    }

    for model_name, model in models.items():

        log_text('logs/question2.txt', f"Performing Grid Search for {model_name}...")

        # Grid Search

        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Training & evaluating the best model

        training_acc = best_model.score(X_train, y_train)
        clf_report, test_acc = evaluate_model(best_model, X_test, y_test)

        log_text('logs/question2.txt', f"Best Parameters for {model_name}: {best_params}")
        log_text('logs/question2.txt', f"Training Accuracy: {training_acc:.4f}")
        log_text('logs/question2.txt', f"Test Accuracy: {test_acc:.4f}")
        log_text('logs/question2.txt', clf_report)
