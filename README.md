# Disaster-Response-Analysis-Using-Machine-Learning

## Introduction
The Project is about a dataset contains real messages that were sent during disaster events. We built a machine learning pipeline to categorize these events so that the message(s) can be sent to an appropriate disaster relief agency.The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

## Libraries Used
- **Pandas**: Fast, powerful, flexible, and easy to use open-source data analysis and manipulation tool. It will help us to view, clean, and apply analysis techniques to the datasets.
- **NumPy**: NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices. It will help us in the mathematical calculation and data analysis.
- **Matplotlib & Seaborn**: These two libraries are so powerful in visualizing data and showing the relationship between features.
- **Scikit-learn**: Fabulous tools for predictive data analysis. They will lead us to create a fantastic Machine Learning model predicting a continuous-valued attribute as we will see later. 
- **Flask**: Flask (source code) is a Python web framework built with a small core and easy-to-extend philosophy.
- **NLTK**: NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.
- **re**: This module provides regular expression matching operations similar to those found in Perl.
- **SQLalchemy & SQLite3**: Python SQL toolkits and object relational Mappers that give application developers the full power and flexibility of SQL.
- **Pickle**: The pickle module implements binary protocols for serializing and de-serializing a Python object structure. We used it to save the ML model as .pkl file.

## Messages and Categories Datasets 
 Messages and Categories datasets are merged to give us the full dataframe that we will clean and use to build the ML model. `messages.csv` contains the messages text for each tweet, and the `categories.csv` contains the messages' related category (Fire, Storm, First Aid, etc...).
 

## Data Extracting, Transforming, and Loading (ETL)
In a Python script, `process_data.py`, we have done a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

## Modeling
In this step, we created a Machine Learning model to predict each message's category. In a Python script, train_classifier.py,we have done a Machine Learning Pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

## Web App
The project set the Machine Learning model in interactive environment so it can be used easly to predict the message related category. The Python script `run.py` contains the backend code, as well as in the template file you can find the html files.

![alt text](https://github.com/ZaidGhazal/Disaster-Response-Using-Machine-Learning/blob/main/WebApp.png?raw=true)
     