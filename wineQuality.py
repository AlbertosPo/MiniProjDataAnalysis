import numpy as np
import torch 
import torch.nn as nn

import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


df = pd.read_csv('winequalityN.csv')

print(df)

## Here our label is quality of wine

# print(df['quality'])
# print(df.head())

### Checking how many nulls have got each column
#print(df.isnull().sum())

def SeparatingByType(dataFrame):
    ### Separating whine and red wine
    # Notice , its working only for classified type column (Check out csv file , 
    # you can easy realise type column is placed sequentially  )

    trueFalse = dataFrame['type'] == 'white'
    countAll = int(  (dataFrame.shape)[0] )
    print('Count all',countAll)
    countOfWhite = int(trueFalse.sum() )
    countOfRed = countAll - countOfWhite
    print("Count of White wines", countOfWhite)
    print("Count of Red wines", countOfRed)

    whiteFun = dataFrame.iloc[:countOfWhite,:]
    redFun = dataFrame.iloc[countOfWhite:,:]

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
    




red,white = SeparatingByType(df)
# print('##########################################################')
# print(white)
# print('##########################################################')
# print(red)

bestWineQuality(red,white,df)
































