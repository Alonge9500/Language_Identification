#Load Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import pickle
import unittest

#Load Pickle Files
with open('../pickles/final_naivebaye_model.pkl','rb') as file:
    classifier = pickle.load(file)

with open('../pickles/tfidf_vectorizer.pkl','rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('../pickles/language_label_mapping.pkl','rb') as file:
    mapping = pickle.load(file)

spanish = ['hola, como estas?', 'Buenos dias', 'Por favor, donde esta','Adios, hasta manana']
german = ['Guten Morgen','Wie geht es ihnen','Wo ist die Toilette?','Auf wiedersehen']
afrikaans = ['goeie more','hoe gaan dit met jou?','waar is die badkamer?','Ek verstaan nie']
afrikaans_with_english = ['how are you doing?','hoe gaan dit met jou?','waar is die badkamer?','Whats up how','Ek verstaan nie']


def testing_model(phrases:list, language:str)-> int:
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


class TestLangDetectionModel(unittest.TestCase):
    
    def test1(self):
        self.assertEqual(testing_model(spanish,'spa'),4)

    def test2(self):
        self.assertEqual(testing_model(german,'deu'),4)

    def test3(self):
        self.assertEqual(testing_model(afrikaans,'afr'),4)

    def test4(self):
        self.assertEqual(testing_model(afrikaans_with_english,'afr'),3)


if __name__ == '__main__':
    unittest.main()
