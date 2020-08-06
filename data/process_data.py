# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:34:37 2020

@author: chaerder
"""

# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Requires & loads two .csv files given the filepaths.
    They both require an id column as key, two merge both files
    OUTPUT: A dataframe consisting of the merged datasets on id
    """ 
    # read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge both loaded datasets
    df_merged = categories.merge(messages, on = 'id')

    # Output
    return df_merged

# messages_filepath = '//cilchscfps21/cilchpersh$/chaerder/udacity/Disaster Response Pipeline/messages.csv'
# categories_filepath = "//cilchscfps21/cilchpersh$/chaerder/udacity/Disaster Response Pipeline/categories.csv"
# test_load = pd.DataFrame()
# test_load = load_data(messages_filepath, categories_filepath)

def clean_data(df):
    """
    After loading the dataset in load_data, the data is cleaned in this function
    (Duplicates dropped) and the categories file fully used by correctly getting
    the message categories information. Categories have after loading two columns:
    ID and categories, whereby categories consists of multiple categories encoded
    the following:
    << categorie_x-1/0 >> whereby 1/0 stands for Category TRUE/FALSE.
    For each ID, we want to have each category as a column.
    OUTPUT: a cleaned dataframe with the columns:
        - id
        - message
        - original
        - genre
        - categorie_n : 1, ..., n   
    """
    # Split the values in the categories column on the ; character so that
    # each value becomes a separate column, as 
    categories = df['categories'].str.split(pat = ';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    
    # This row is used to extract a list of new column names for categories.
    # The second to last character of each string is sliced using a lambda fct.
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

# test_clean = clean_data(test_load)

def save_data(df, database_filename):
    """
    The provided dataframe is written to a SQL DB given the provided database path
    OUTPUT: NO OUTPUT
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    print("Dataframe has been written to the SQL DB")

def main():
    if len(sys.argv) == 4:
        
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning & formatting the data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('Cleaned & formatted data was saved to the provided SQL database.')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline