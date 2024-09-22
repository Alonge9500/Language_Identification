#Load Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
import pickle
import time
from model_building import train_features,test_features,val_features,train_label,test_label,val_label

def grid_search_tuning(train_features: pd.Series,
                       test_features: pd.Series,
                       val_features: pd.Series,
                       train_label: pd.Series,
                       test_label: pd.Series,
                       val_label: pd.Series):
    """
    Perform hyperparameter tuning for Naive Bayes, Logistic Regression, and Random Forest models using GridSearchCV.
    
    Parameters:
    - train_features: Training features (pd.Series)
    - train_label: Training labels (pd.Series)
    - test_features: Testing features (pd.Series)
    - test_labels: Testing Labels (pd.Series)
    - val_features: Validation features (pd.Series)
    - val_label: Validation Labels (pd.Series)
    Returns:
    - Best hyperparameters for each model
    """
    
    # Naive Bayes Hyperparameter Tuning
    naive_bayes = MultinomialNB()
    naive_bayes_params = {
        'alpha': [0.1, 0.5, 1.0],
        'fit_prior': [True, False] # hyperparameter values for Naive Bayes
    }
    print('Starting HyperParameter Tuning')
    start_time = time.time()
    naive_bayes_grid = GridSearchCV(estimator=naive_bayes,
                                    param_grid=naive_bayes_params,
                                    cv=5,
                                    scoring='f1_weighted')
    
    naive_bayes_grid.fit(train_features, train_label)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Grid Search HyperParameter Tuning Completed time taken is {time_taken:.2f} seconds')
    naive_bayes_grid_predictions = naive_bayes_grid.predict(test_features)
    print(f'Best Params for Naive Bayes: {naive_bayes_grid.best_params_}')

    print('Naive Bayes Model Grid Search Evaluation On Testing Subset:')
    print(f'F1-Score: {f1_score(test_label, naive_bayes_grid_predictions, average="weighted"):.4f}')
    print(f'Recall: {recall_score(test_label, naive_bayes_grid_predictions, average="weighted"):.4f}')
    print(f'Precision: {precision_score(test_label, naive_bayes_grid_predictions, average="weighted"):.4f}')
    print('Classification Report:\n', classification_report(test_label, naive_bayes_grid_predictions))

    
    naive_bayes_grid_predictions_validation = naive_bayes_grid.predict(val_features)
    print('Naive Bayes Model Grid Search Evaluation On Validation Subset:')
    print(f'F1-Score: {f1_score(val_label, naive_bayes_grid_predictions_validation, average="weighted"):.4f}')
    print(f'Recall: {recall_score(val_label, naive_bayes_grid_predictions_validation, average="weighted"):.4f}')
    print(f'Precision: {precision_score(val_label, naive_bayes_grid_predictions_validation, average="weighted"):.4f}')
    print('Classification Report:\n', classification_report(val_label, naive_bayes_grid_predictions_validation))

    with open('pickles/final_naivebaye_model.pkl','wb') as file:
        pickle.dump(naive_bayes_grid, file)

    print('Model Saved To Pickle Folder')
    
   
grid_search_tuning(train_features, test_features, val_features,train_label, test_label,val_label)