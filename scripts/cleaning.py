#Load Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



main_data = pd.read_csv('../data.csv')

def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters: dataframe(pd.DataFrame): Main Downloaded Data

    Return: pd.DataFrame (A Clean Version of Data Frame)
    
    """
    dataframe['class'] = dataframe['class'].apply(lambda x: x[:-1])

    print('Cleaning Completed')
    dataframe.to_csv('../data_folder/clean_data.csv', index=False)

clean_data(main_data)