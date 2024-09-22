#Load Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
import pickle

data = pd.read_csv('../data_folder/clean_data.csv')

def extract_selected_language(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters: dataframe(pd.DataFrame) -> Main Data Frame containing all Data 

    Returns: pd.DataFrame -> A Dataframe containing only Afrikaans, Spanish, German, Alemannic German

    """
    extracted_data = dataframe[(dataframe['class'] == 'afr') | (dataframe['class'] == 'spa') | (dataframe['class'] == 'als') | (dataframe['class'] == 'deu')]
    print('Succesfully Extracted Data')

    extracted_data.to_csv('../data_folder/extracted_data.csv', index=False)
    print('Succesfully Saved To Csv')
    return extracted_data

#Extracting and Saving Data
extracted_data = extract_selected_language(data)


def feature_extraction_et_label_encoding(dataframe: pd.DataFrame) -> list:
    """
    Parameters:  dataframe: pd.DataFrame -> Data frame of selected language

    Return: A list [text_vectors, language_label, language_label_mapping]

    Receive a Dataframe and perform the following operations
    * Split to Features and Labels
    * Perform Labeel Encoding on the Label Section
    * Perform Feature Extarction usinf TfIDF on the feature Section
    * Extract Label Mapping
    * Return A list Containing The Vectors, Labels and Label Mapping
    
    """

    # Split Data to Texts and Labels
    texts = dataframe.text
    language_label = dataframe['class']
    
    #Encoding
    label_encoder = LabelEncoder() # initialize Encoder
    language_label= label_encoder.fit_transform(language_label)
    language_label_mapping = dict(zip(label_encoder.classes_,
                                     label_encoder.transform(label_encoder.classes_)))

    print('Label Encoding Completeed')

    # Save Encoding Dictionary
    with open('../pickles/language_label_mapping.pkl','wb') as file0:
        pickle.dump(language_label_mapping, file0)
    
    #Vectorization
    tfidf_vectorizer = TfidfVectorizer() # Initialize the Vectorizer

    tfidf_text_vectors = tfidf_vectorizer.fit_transform(texts).toarray()
    print('Vectorization Complete')
    
    # Save Vectorizer
    with open('../pickles/tfidf_vectorizer.pkl','wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    print('Vectorizer Saved to pickles folder')
    
    print('Encoding And Preprocessing Completed')
    return [tfidf_text_vectors,language_label,language_label_mapping]


#Encoding and Vectorizing
vectors, label, mapping = feature_extraction_et_label_encoding(extracted_data)
