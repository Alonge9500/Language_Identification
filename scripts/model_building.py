#Load Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
import pickle
import time
from preprocess import vectors, label, mapping

def split_data(vectors: pd.Series, label: pd.Series) -> tuple:
    """
    Parameters vectors(pd.Series) label(pd.Series)

    Return: A tupple of the split features

    Split Data into Train and test features with 30 Percent to test and 70 to training
    
    """
    train_features, test_val_features, train_label, test_val_label = train_test_split(vectors,
                                                                              label,
                                                                              test_size=0.30, 
                                                                              random_state=3)

    test_features, val_features, test_label, val_label = train_test_split(test_val_features,
                                                                              test_val_label,
                                                                              test_size=0.33, 
                                                                              random_state=3)
    print('Data Splitted successfully to Training, Testing and Validation set')
    return train_features,test_features,val_features,train_label,test_label,val_label

def train_models(train_features: pd.Series,
                 test_features: pd.Series,
                 train_label: pd.Series,
                 test_label: pd.Series):

    """
    Parameters : 
    train_features -> features to be use in training (pd.Series)
    test_features -> features for training (pd.Series)
    train_labels -> training labels (pd.Series)
    test_labels -> testing labels (pd.Series)
    Return A tupple for the prediction of all Models

    Train 3 Language Identification Models
    
    """
    #Models Initialization
    naive_bayes_model = MultinomialNB() # Naive Bayes
    logistic_regression_model = LogisticRegression() # Logistic Regression
    random_forest_classifier_model = RandomForestClassifier() # Random Forest Classifier
    print('Models Initialization Completed')
    

    #Models Training
    start_time = time.time()
    naive_bayes_model.fit(train_features, train_label)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Naive Bayes Completed time taken is {time_taken:.2f} seconds')
    
    start_time = time.time()
    logistic_regression_model.fit(train_features, train_label)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Logistic Regression Completed time taken is {time_taken:.2f} seconds')


    start_time = time.time()
    random_forest_classifier_model.fit(train_features, train_label)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Random Forest Completed time taken is {time_taken:.2f} seconds')
    
    print('Models Training Completed')

    #Models Predictions
    naive_bayes_model_prediction = naive_bayes_model.predict(test_features)
    logistic_regression_model_prediction = logistic_regression_model.predict(test_features)
    random_forest_classifier_model_prediction = random_forest_classifier_model.predict(test_features)
    print('Models Predictions Complete')


    return naive_bayes_model_prediction, logistic_regression_model_prediction, random_forest_classifier_model_prediction


    


def evaluate_models(test_label: pd.Series, 
                    naive_bayes_prediction: pd.Series, 
                    logistic_regression_prediction: pd.Series, 
                    random_forest_prediction: pd.Series):
    
    """
    Parameters:
    - test_label: True labels for the test set (pd.Series)
    - naive_bayes_prediction: Predictions from the Naive Bayes model (pd.Series)
    - logistic_regression_prediction: Predictions from the Logistic Regression model (pd.Series)
    - random_forest_prediction: Predictions from the Random Forest model (pd.Series)
    
    Return:
    - Prints evaluation metrics including F1-score, Recall, Precision, and Classification Report for all models.
    """
    
    # Model Evaluation: Naive Bayes
    print('Naive Bayes Model Evaluation:')
    print(f'F1-Score: {f1_score(test_label, naive_bayes_prediction, average="weighted"):.4f}')
    print(f'Recall: {recall_score(test_label, naive_bayes_prediction, average="weighted"):.4f}')
    print(f'Precision: {precision_score(test_label, naive_bayes_prediction, average="weighted"):.4f}')
    print('Classification Report:\n', classification_report(test_label, naive_bayes_prediction))
    
    print('-' * 50)
    
    # Model Evaluation: Logistic Regression
    print('Logistic Regression Model Evaluation:')
    print(f'F1-Score: {f1_score(test_label, logistic_regression_prediction, average="weighted"):.4f}')
    print(f'Recall: {recall_score(test_label, logistic_regression_prediction, average="weighted"):.4f}')
    print(f'Precision: {precision_score(test_label, logistic_regression_prediction, average="weighted"):.4f}')
    print('Classification Report:\n', classification_report(test_label, logistic_regression_prediction))
    
    print('-' * 50)
    
    # Model Evaluation: Random Forest
    print('Random Forest Model Evaluation:')
    print(f'F1-Score: {f1_score(test_label, random_forest_prediction, average="weighted"):.4f}')
    print(f'Recall: {recall_score(test_label, random_forest_prediction, average="weighted"):.4f}')
    print(f'Precision: {precision_score(test_label, random_forest_prediction, average="weighted"):.4f}')
    print('Classification Report:\n', classification_report(test_label, random_forest_prediction))

    print('Evaluation Completed')

#Split Data
train_features,test_features,val_features,train_label,test_label,val_label = split_data(vectors,label)
#Train Models
naive_bayes_pred, logistic_regression_pred, random_forest_pred = train_models(train_features,
                                                                              test_features,
                                                                              train_label,
                                                                              test_label)
# Call evaluate function
evaluate_models(test_label, naive_bayes_pred, logistic_regression_pred, random_forest_pred)