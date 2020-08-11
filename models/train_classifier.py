# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 07:59:44 2020

@author: chaerder
"""
import sys
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sqlalchemy import create_engine
import re

# testing different statistical learning methods
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, f1_score, fbeta_score, classification_report

import pickle

import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    """
    Loads a SQLite .db file and extracts the model data.
    
    OUTPUT: The model data & category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con = engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    # Dont need the engine anymore
    engine.dispose()

    return X, y, category_names

# test_load = load_data('C:/Users/Clemens/Documents/DisasterResponsePipeline/data/DisasterResponse.db')

def tokenize(text):
    """
    Normalizes the text and removes URL & normal regular expressions.
    The text is then tokenized. Then, punctation & english stopwords are removed.
    Finally, the words are lemmatized to better analyze the words.
    
    OUTPUT: list of lemmatized tokens free from punctuation.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # normalize text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize punctuation
    tokenice = RegexpTokenizer(r'\w+')
    tokens = tokenice.tokenize(text)

    # Remove english stopwords
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]

    # Lemmatize the tokens without stopwords
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    The model pipeline for a ulticategorical classification using
    a RandomForest classifier. As input, text is vectorized
    and then tf-idf-transformed.
    
    OUTPUT: A model pipeline which can be used for model training.
    """
    
    pipeline = Pipeline([
      ('vect', CountVectorizer(tokenizer = tokenize)),
      ('tfidf', TfidfTransformer()),
      ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Number of trees in random forest
    n_estimators = [20, 50, 100]
    # Maximum number of levels in tree
    max_depth = [10, 30, 60]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # hyper-parameter grid
    parameters = {
        'clf__estimator__n_estimators': n_estimators,
        'clf__estimator__max_depth': max_depth,
        'clf__estimator__min_samples_split': min_samples_split,
        'clf__estimator__min_samples_leaf': min_samples_leaf,
    }
    
    # GridSearch best parameter setting
    model = GridSearchCV(estimator = pipeline,
                         param_grid = parameters,
                         verbose = 10,
                         cv = 2)

    # create model
    return model


def display_results(y_test, y_pred, model):
    """
    The function uses the model and predictions to generate the model output.
    
    OUTPUT: A model output dataframe.
    """
    model_output = pd.DataFrame(columns = ['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for colnm in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[colnm], y_pred[:,num], average = 'weighted')
        model_output.at[num + 1, 'Category'] = colnm
        model_output.at[num + 1, 'f_score'] = f_score
        model_output.at[num + 1, 'precision'] = precision
        model_output.at[num + 1, 'recall'] = recall
        num += 1
    print('Aggregated f_score:', model_output['f_score'].mean())
    print('Aggregated precision:', model_output['precision'].mean())
    print('Aggregated recall:', model_output['recall'].mean())
    print('Accuracy:', np.mean(y_test.values == y_pred))
    print("\nBest Parameters:", model.best_params_)
    
    return model_output

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the provided model given a test-dataset
    
    OUTPUT: The models prediction
    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    results = display_results(y_test, y_pred, model)
    print(results)
        
def save_model(model, model_filepath):
    """
    Saves a model to a provided location as pickl file.
    
    OUTPUT: a model as pickl file.
    """
    pickl_con = open(model_filepath, 'wb')
    pickle.dump(model, pickl_con)
    pickl_con.close()


def main():
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
