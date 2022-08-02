import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    loads messages.csv and categories.csv, merges them and returns the
    merged dataframe
    
    input:
            messages_filepath: filepath of messages.csv
            categories_filepath: filepath of categories.csv
    output:
            df: dataframe merged on the "id" column
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on=["id"])
    return df


def clean_data(df):
    '''
    cleans and processes the "categories" column in df to individual 
    columns with a numeric value
    
    input:
            df: dataframe with "categories" column to be cleaned and processed
    output:
            df: dataframe with cleaned and processed "categories" column
    '''
    
    categories = df.categories.str.split(pat=';',expand=True)
    
    row = categories.iloc[0]
    
    category_colnames = []
    for x in row:
        category_colnames.append(x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1)
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    
    df.drop(['categories'], axis=1, inplace = True)
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace = True)
    
    return df
    
    
def save_data(df, database_filename):
    '''
    saves cleaned dataframe to sqlite database
    
    input:
            df: clean dataframe
            database_filename: file name of database to save to
    '''
    engine = create_engine(database_filename)
    df.to_sql(df, engine, index=False, if_exists = 'replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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
