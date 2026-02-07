
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd



import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset , DataLoader,TensorDataset
from sklearn.model_selection import train_test_split # Split the dataset (train, validation, test)



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

def SwitchColumnsFirToFin(df_in):
    ### Have to replace white and red string values to (0,1)
    df_in.type = df_in.type.map({'white':0 , 'red':1})


    list_of_titles = df_in.columns.values.tolist()

    ### Quick way to switch strings in list
    list_of_titles[0], list_of_titles[-1] = list_of_titles[-1], list_of_titles[0]


    df_in = df_in.reindex(columns = list_of_titles)
    return df_in 



def FunctionZscore(data_in):

    ss = StandardScaler()
    scaled = ss.fit_transform(data_in)

    res = pd.DataFrame(scaled , columns = data_in.columns)
    return res



def SeparatingDataFromLabels(df_in):
    labels = df_in["type"]
    df_in.drop(['type'],axis = 1,inplace = True)
    data = df_in.copy()
    return data,labels


def TrainTest(X_in,y_in):


    ### Converting dataframe to tensor 
    X_in = torch.tensor(X_in.values , dtype = torch.float32)
    y_in = torch.tensor(y_in.values , dtype = torch.float32)

    # First attempt without shuffling the data
    X_train , X_test , y_train , y_test = train_test_split(X_in,y_in,test_size = 0.2 ,shuffle = False) 

    return X_train,X_test , y_train , y_test 


class NeuralNet(nn.Module):
    def __init__(self,columns):
        super().__init__()

        ### Inpute layer
        self.input = nn.Linear(columns,15)

        ### hidden layers
        self.fc1 = nn.Linear(15,15)
        self.fc2 = nn.Linear(15,15)

        ### output layer
        self.output = nn.Linear(15,1)

    def forward(self,x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)





#red,white = SeparatingByType(df)
# print('##########################################################')
# print(white)
# print('##########################################################')
# print(red)
#bestWineQuality(red,white,df)

df = CleaningNullAndReset(df)

df = SwitchColumnsFirToFin(df)

### To keep df dataframe untouchable
dfToSplit = df.copy()

print(df)
dfToSplit = FunctionZscore(dfToSplit)
print(dfToSplit)


# Here, we get type column as labels separate than rest dataframe . 
# Rest dataframe saved in data
data , labels = SeparatingDataFromLabels(dfToSplit)



X_train,X_test , y_train , y_test = TrainTest(data,labels)



rows , columns = X_train.shape
print(rows,columns)

# start with a fresh network
net = NeuralNet(columns)
optimizer = torch.optim.Adam(net.parameters(),lr=.0001)
lossfun = nn.BCEWithLogitsLoss() # try with different loss function


numEpochs = 100

for epochi in range(numEpochs):


    y_pred = net(X_train)
    y_train = y_train.reshape(-1,1)  
    loss_train = lossfun(y_pred,y_train)


    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

print("OKEY")



















