import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from data_ingestion import load_data
from zenml import pipeline, step


import warnings
warnings.filterwarnings('ignore')

df = load_data('card_transdata.csv')

@step
def eda_step(df):
    #exploring the data
    print(df.head())

    print(df.tail())

    print(df.info())


    #Droping duplicate values
    df.drop_duplicates(inplace=True)

    #Exploratory Data Analysis

    print(df.describe())

    # Verifying the Skewness

    df.skew()

    #checking for null values in each column

    df.isnull().sum()
    '''
    #Visualization of data

    # Set up subplots for all visualizations
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))


    df[['distance_from_home', 'distance_from_last_transaction']].plot(ax = axs[0, 0])
    axs[0, 0].set_title('Distance Features Over Transactions')
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('Distance')

    # Boxplot
    axs[0, 1].boxplot(df)
    axs[0, 1].set_title('Boxplot to identify outliers')
    axs[0, 1].set_xlabel('Variable Position')
    axs[0, 1].set_ylabel('Values')

    #Histogram

    columns = ['repeat_retailer', 'used_chip',
           'used_pin_number', 'online_order', 'fraud']

    axs[1, 0].hist(df[columns], histtype='bar', bins = 15, label = columns)
    axs[1, 0].legend()
    axs[1, 0].set_xticks([0,1])
    axs[1, 0].set_xlabel('Values')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Distribution of Binary Features')

    # Correlation Heatmap

    cor_coef = df.corr()

    sns.heatmap(cor_coef,annot=True, cmap='coolwarm', ax = axs[1, 1])
    axs[1, 1].set_title('Correlation Heatmap')

    plt.tight_layout()
    #plt.show()
    fig.savefig('Credit_Card_Fraud_Detection_EDA.png')'''

    return df



