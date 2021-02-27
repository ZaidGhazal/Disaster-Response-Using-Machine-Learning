# import libraries
import sys
import pandas as pd
import pickle
import sqlite3
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
stop_words = nltk.corpus.stopwords.words('english')

def load_data(database_filepath):
    ''' Load data from the SQL database given in the path, then create X and Y variables for messages (X) and target classes (Y).

        Args:
            database_filepath: Database file path

        return:
            X: messages as numpy array
            Y: targets as numpy array (36 targets, some have multiclasses)   
            category_names: targets names list  
    '''
    # load data from database
    con = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM data_table',con)

    #create X as messages (numpy array) and Y as targets (numpy array has 36 targets, some have multiclasses)
    X = df['message'].values
    Y = df.drop(columns=['id','message', 'original', 'genre']).values

    # create the targets names list  
    category_names = df.drop(columns=['id','message', 'original', 'genre']).columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    ''' Prepare the text (message) to be ready for training the model

        Args:
            text: the message text
        
        return: The message after tokenization    
    
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens
    


def build_model():
    ''' Create the model pipline

        Args:
            None
        
        return: 
            The created model
    
    '''

    # Create the pipline 
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', ExtraTreesClassifier(n_jobs=-1))
    ])

   # Define the parameters for the GridsearchCV 
    parameters = {
    'clf__n_estimators': [10,20,50,100]
    }

    # Define the GridSearchCV and its arguments
    scorer = make_scorer(score_func= f1_score, average='weighted', zero_division=1)
    cv = GridSearchCV(pipeline,param_grid=parameters, scoring=scorer, verbose=1, cv=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluate the model performance after fitting to training data. Also print the performance 
         report for each target.

        Args:
            model: The fitted model to be evaluated
            X_test: Test dataset
            Y_test: Test dataset's targets
            category_names: List of the targets names
        
        return: 
            None
    
    '''

    # Predict the targets using the test data
    y_pred = model.predict(X_test)
    
    # Print each target's performance report
    for i,column in enumerate(category_names):
        target_name = [column]
        print('REPORT for Column: ' + column)
        print(classification_report(Y_test[:,i], y_pred[:,i], zero_division=1))


def save_model(model, model_filepath):
    ''' Save the model as pickel file.

        Args:
            model: The fitted model to be saved
            model_filepath: Path to save the model
        
        return: 
            None
    
    '''

    # Save the model to the specified path
    filename = model_filepath
    pickle.dump(model.best_estimator_, open(filename, 'wb'))


def main():
    ''' Execute the full process. Load data, build a model, fit and evaluate the model, and
        save the model as pickel fild 

        Args:
            None
        return: 
            None
    
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print(model.best_estimator_)
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