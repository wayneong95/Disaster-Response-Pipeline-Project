import nltk
nltk.download(['punkt', 'wordnet'])

import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    loads a dataframe from database_filepath and extracts the X and Y variables 
    as well as the labels
    
    input:
            database_filename: file name of database to load from
    
    output:
            X (dataframe): X variables
            Y (dataframe): Y variables
            category_names: list of category names
    '''
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('df', engine)
    
    X = df['message'].values
    Y = df.drop(['id', 'message','original','genre'], axis=1).values.astype(int)
    category_names = df.drop(['id', 'message','original','genre'], axis=1).columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    returns tokenized, cleaned and lemmatized text
    
    input:
            text: text to be tokenized, cleaned and lemmatized
    
    output:
            clean_tokens: tokenized, cleaned and lemmatized text
    
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    builds and returns machine learning pipeline
    
    output:
            pipeline: machine learning pipeline
    '''
    
    knn = KNeighborsClassifier()
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(knn)),
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    prints the classification report for each target column
    
    input:
            model: model to be evaluated
            X_test: test set used to predict
            Y_test: actual target values
            category_names: names of target columns
    '''
    Y_pred = model.predict(X_test)
    
    for col in range(Y_test.shape[1]):
        print(classification_report(Y_test[:,col],Y_pred[:,col]))


def save_model(model, model_filepath):
     '''
    saves trained model to a pickle file
    
    input:
            model: trained model
            model_filepath: filepath to save model to
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
