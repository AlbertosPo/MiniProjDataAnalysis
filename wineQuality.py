import numpy as np
import torch 
import torch.nn as nn

import pandas as pd



import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


dataframe = pd.read_csv('winequalityN.csv')
df = dataframe.copy() ### To keep original dataframe in 'dataframe'

#print(df)

## Here our label is quality of wine

# print(df['quality'])
# print(df.head())

### Checking how many nulls have got each column
#print(df.isnull().sum())

def SeparatingByType(dataFrame_in):
    ### Separating whine and red wine
    # Notice , its working only for classified type column (Check out csv file , 
    # you can easy realise type column is placed sequentially  )

    trueFalse = dataFrame_in['type'] == 'white'
    countAll = int(  (dataFrame_in.shape)[0] )
    print('Count all',countAll)
    countOfWhite = int(trueFalse.sum() )
    countOfRed = countAll - countOfWhite
    print("Count of White wines", countOfWhite)
    print("Count of Red wines", countOfRed)

    whiteFun = dataFrame_in.iloc[:countOfWhite,:]
    redFun = dataFrame_in.iloc[countOfWhite:,:]

    print('ByeBye')
    return redFun,whiteFun

def bestWineQuality(red_in,white_in,df_in):
    ### Notice , which type of wine's quality is better
    countRed = int(red_in.shape[0])
    countWhite = int(white_in.shape[0])
    print('red',countRed)
    print('white',countWhite)
    meanOfWhite = df_in.loc[:countWhite,'quality'].mean()
    meanOfRed = df_in.loc[countWhite:,'quality'].mean()
    print("Mean of white wine quality",meanOfWhite)
    print("Mean of red wine quality",meanOfRed)
    
### This function just clean null values and reset the numerical order of rows ,that dataframe has.
def CleaningNullAndReset(df_in):
    df_in.dropna(inplace=True)
    df_in = df_in.reset_index(drop=True) # reseting rows of dataframe.
    return df_in



#red,white = SeparatingByType(df)
# print('##########################################################')
# print(white)
# print('##########################################################')
# print(red)
#bestWineQuality(red,white,df)

df = CleaningNullAndReset(df)


### Have to replace white and red string values to (0,1)
df.type = df.type.map({'white':0 , 'red':1})

column_names = ['type','quality']

list_of_titles = df.columns.values.tolist()
print(list_of_titles)


print(df)

























