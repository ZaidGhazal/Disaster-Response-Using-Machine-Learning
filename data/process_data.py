# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    ''' Load messages and categories the data from the specified file pathes
        Args:
            messages_filepath: messages dataset (.csv) file path
            categories_filepath: categories dataset (.csv) file path
        return:
            df: Pandas dataframe contains the messages and categories datasets merged by columns    
    '''
    #Load the messages and categories datasets and join them on id column
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='inner')

    return df


def clean_data(df):
    ''' Clean the specified dataframe 
        Args:
            df: Pandas dataframe contains the messages and categories datasets merged by columns
        return:
            df: Cleaned Pandas dataframe contains the messages and 36-categories representing columns    
    '''
    
    categories = df['categories'].str.split(pat=';', expand=True)
    
    
    # Save the new columns names  
    row = categories.loc[0]
    category_colnames = categories.loc[0].apply(lambda x: x.split('-')[0]).tolist()
    categories.columns = category_colnames
    
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast='integer')
    
    
    # Drop the original catigories column and concat the 36 categories columns with the messages dataframe
    # Also drop the duplicated records after concating
    df = df.drop(columns='categories')
    df = pd.concat([df,categories], axis=1)
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    df = df[df.related != 2]
    return df

def save_data(df, database_filename):
    ''' Save the specified file path 
        Args:
            df: Cleaned Pandas dataframe contains the messages and 36-categories representing columns    
            database_filename: SQL database file path
        return:
            None
    '''
    
    # Save the cleaned dataset as SQL (.db) database 
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('data_table', engine, index=False, if_exists='replace')  


def main():
    ''' Load data, clean the loaded data, and save the cleaned dataset as .db file (Execute the program) 
        Args:
            None
        return:    
            None
    '''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df.copy())
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()