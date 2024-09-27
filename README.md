# Language_Identification

## Project Structure

|──data_folder
|   |──extracted_data.csv
|──pickles
|   |──language_label_mapping.pkl
|   |──tfidf_vectorizer.pkl
|   |──final_naivebayes_model.pkl
|──scripts
|   |──__init__.py
|   |──cleaning.py
|   |──preprocess.py
|   |──model_building.py
|   |──hyperparameter_tuning.py
|   |──testing1.py
|   |──testing2.py
|──language_identification_notebook.ipynb
|──data.csv
|──language_identification_app.py
|──README.md
|──requirements.txt
|


## About Data
* Data Name : Wili 2018
* Source: Kaggle [https://www.kaggle.com/datasets/sharansmenon/wili-2018?select=data.csv]
* The main language data. Contains about 200k instances for 235 languages

# STEPS
### Load Data
### Data Cleaning
* Remove Escape strings at the end of the Data

### Data Preprocessing
* Select only data label to Afrikaans, Spanish, German and Alemannic German only
* Label Encoding using LabelEncoder Class in sklearn.preprocessing module
* Feature Extraction (TF-IDF- Term Frequency-Inverse Document Frequency)

### Model Training and Evaluation
* Split Data
  - Splitted Data to Train, Test and Validation set 70%, 20% and 10% Respectively
* Train Data on 3 Model
  * Logistic Regression -> F1-Score: 0.9842 -> Recall: 0.9842 -> Precision: 0.9843 -> Time Taken: 292 Seconds
  * Naive Bayes -> F1-Score: 0.9883 -> Recall: 0.9883 -> Precision: 0.9888 -> Time Taken: 19 seconds
  * Random Forest Classifier -> F1-Score: 0.9875 -> Recall: 0.9875 -> Precision: 0.9879 -> Time Taken: 75 seconds
- Base on the Reason Below we will be selecting Naive Bayes for this Project
  - *Results*: Naive bayes happen to return the best result for F1, Recall and Presision
  - *Training Time*: Naive Bayes took lesser time to train in comparison to other models 
  - *Model Size*: Naive Bayes tend to occupy lesser space in the memory
  - *Versertility*: Naive Bayes is Also popular for Text classification problems

### Hyperparameter Tuning
To further increase the metrics result of our naive bayes model we use GridSearch CV(Cross Validation)
To further improve the model and we got the following Result

* Old Result -> F1-Score: 0.9883 -> Recall: 0.9883 -> Precision: 0.9888
* New Result -> F1-Score: 0.9901 -> Recall: 0.9900 -> Precision: 0.9904
* Validation Set Result on Grid seach model is 100% for F1 score, Presision and Recall
- Comparing the two results above we notice a significant improvement
- Best Params for Naive Bayes: {'alpha': 0.1, 'fit_prior': True}


### Deployment
- The Final Naive bayes was serialized using pickle and the model was deploy using streamlit
#### Running The App
To Run the app use the command below
  
`streamlit run language_identification_app.py`

### Unit Test
* There are two testing file in the scripts folder, testing1.py and testing2.py
* testing1.py was design with unittest approach
  To run testing 1 use the command below
  `python run testing1.py`
* testing2.py was design with pytest approach
  `pytest testing2.py` provided you have pytest installed

#### Video Walkthrorugh of deployment
<video src='https://github.com/user-attachments/assets/42e8c4b7-24d7-43d1-87c3-d0dbfa6d180e'></video>



