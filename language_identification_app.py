#Load Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import pickle
import streamlit as st

#Set page config

st.set_page_config(
    page_title = "Language Identification APP",
    page_icon = ":sparkles",
    layout = "centered",
    initial_sidebar_state = "expanded"
)
#Load Pickle Files
with open('pickles/final_naivebaye_model.pkl','rb') as file:
    classifier = pickle.load(file) # Classifier (Naive Bayes)

with open('pickles/tfidf_vectorizer.pkl','rb') as file:
    tfidf_vectorizer = pickle.load(file) 

with open('pickles/language_label_mapping.pkl','rb') as file:
    mapping = pickle.load(file) # Label Mappings

def main():
    st.title("Multilingual Identifier")
    st.markdown("*Instruction*:The Language Identification APP(Multilingual Identifier)\
                only works for 3 languages Afrikaans, Spanish and German\
                For both (Alemannic German and Main German)\
                \n* Enter a phrase in any of the above listed language and the system will identify it\
                \n* Example \
                \n \t -(`hola, como estas?` -> Spanish)\
                \n \t -(`Wie geht es ihnen` -> German)\
                \n \t -(`hoe gaan dit met jou?` -> Afrikaans)")

    #Collect User Input
    user_input = st.text_area("Enter Text:")

    if st.button("Identify"):
        if user_input.strip() != "":
            vectors = tfidf_vectorizer.transform([user_input]).toarray()
            prediction = list(classifier.predict_proba(vectors)[0])
            if max(prediction) < 0.5:
                st.warning("The Inputed pharase is neither of the supported languages,\
                            Go through the instructions again")
            else:
                prediction = classifier.predict(vectors)[0]
                if prediction == 0:
                    st.success('The inputed Phrase language is -> Afrikaans')
                elif prediction == 1:
                    st.success('The inputed Phrase language is -> Alemannic German')
                elif prediction == 2:
                    st.success('The inputed Phrase language is -> German')
                elif prediction == 3:
                    st.success('The inputed Phrase language is ->  Spanish')

if __name__ == '__main__':
    main()
