#Load Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import pickle


#Load Pickle Files
with open('../pickles/final_naivebaye_model.pkl','rb') as file:
    classifier = pickle.load(file)

with open('../pickles/tfidf_vectorizer.pkl','rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('../pickles/language_label_mapping.pkl','rb') as file:
    mapping = pickle.load(file)

spanish = ['Buenos dias','Por favor','Hasta Luego']
spanish_with_yoruba = ['Buenos dias','e ku ojumo','e kaaro o']
german = ['Bis spater','Ich verstehe nicht','Entschuldigung']
afrikaans = ['ek verstaan nie','Verskoon my','Sien jou later']


def model_check(phrases:list, language:str)-> int:
    """
    parameters
    phrases:list -> List of phrases to be detected
    language: str -> Expected Language

    return
    SCore

    Test to detect a list of various phrases in a specify language and try to detect if the model get it all right
    Return a score for the number the model got right
    
    """
    score = 0

    for phrase in phrases:
        vectors = tfidf_vectorizer.transform([phrase]).toarray()
        prediction = classifier.predict(vectors)[0]
        
        if mapping[language] == prediction:
            score += 1
        else:
            continue

    return score

class TestLanguageIdentifier:
    
    def test1(self):
        assert model_check(spanish,'spa') == 3
    
    def test2(self):
        assert model_check(german,'deu') == 3
    
    def test3(self):
        assert model_check(afrikaans,'afr') == 3
    
    def test4(self):
        assert model_check(spanish_with_yoruba,'spa') == 1


