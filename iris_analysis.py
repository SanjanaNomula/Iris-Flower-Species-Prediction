import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import requests
import io
import os

# 1. Data Acquisition and Exploration
def load_data():
    url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_eda(df):
    print("\n--- Basic EDA ---")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nClass Distribution:")
    print(df['species'].value_counts())
    
    # Pairplot
    sns.pairplot(df, hue='species')
    plt.title("Pairplot of Iris Dataset")
    plt.savefig('eda_pairplot.png')
    plt.close()

# 2. Preprocessing
def preprocess_data(df):
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le

# 3. Model Implementation and Comparative Analysis
def train_and_evaluate(X_train, X_test, y_train, y_test, le):
    results = []
    
    models = {
        'k-NN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l2'] # lbfgs only supports l2 or None
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        }
    }
    
    best_estimators = {}

    for name, config in models.items():
        print(f"\nTraining {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_estimators[name] = best_model
        
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Best Params: {grid.best_params_}")
        print(f"Accuracy: {acc:.4f}")
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'Best Params': str(grid.best_params_)
        })
        
    return pd.DataFrame(results), best_estimators

def visualize_results(results_df):
    print("\n--- Comparative Analysis ---")
    print(results_df)
    
    # Melt dataframe for plotting
    melted_df = results_df.melt(id_vars=['Model', 'Best Params'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df)
    plt.title("Model Comparison")
    plt.ylim(0.8, 1.0) # Zoom in as accuracy is likely high
    plt.ylabel("Score")
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    df = load_data()
    if df is not None:
        perform_eda(df)
        X_train, X_test, y_train, y_test, le = preprocess_data(df)
        results_df, best_models = train_and_evaluate(X_train, X_test, y_train, y_test, le)
        visualize_results(results_df)

if __name__ == "__main__":
    main()
