import pandas as pd

# libraries for visualizing the data
import matplotlib.pyplot as plt
import seaborn as sns

### Libraries to convert pandas dataframe to tensor matrix with float values
### Also, these libraries will help to train neural network
import torch 
import torch.nn as nn






### Let's read our Iris csv file

dataframe = pd.read_csv('Iris.csv')

# Of course, Have to check for Null value 
# print(dataframe.isnull().sum() ) # Any null value , perfect condition. Otherwise, maybe will be good idea 
# just to remove all rows with null values 


### Have got 3 different species so we expecting 3 bar
# sns.countplot(x='Species',data = dataframe)
# plt.show()


def PetalRelationship(dataframe_in):
    ### In this way we can see relationship of three Species 
    ## by two of characteristics PetalLength and PetalWidth centimeter.
    ## When we visualizing it all is clear
    sns.scatterplot(x = 'PetalLengthCm',y = 'PetalWidthCm' , 
                    hue = 'Species' ,data = dataframe_in)
    plt.show()

def SepalRelationship(dataframe_in):
### In the same way we can see relationship between SepalLength and SepalWidth

    sns.scatterplot(x = 'SepalLengthCm',y = 'SepalWidthCm' , 
                    hue = 'Species' ,data = dataframe_in)
    plt.show()


### Visualize regarding the Species of Iris and count all different cases including in data 
def SpeciesCountAndPlot(dataframe_in):
    print(dataframe_in.value_counts("Species") )
    sns.countplot(x = 'Species',data=dataframe_in)
    plt.show()

#SpeciesCountAndPlot(dataframe)



### Visualize the petal and sepal varieties of Iris 
def VarietiesBars(dataframe_in):
    fig,axs = plt.subplots(2,2,figsize=(10,8))

    axs[0,0].set_title("Sepal Length")
    axs[0,0].hist(dataframe_in['SepalLengthCm'],bins=7)
    
    axs[0,1].set_title("Sepal Width")
    axs[0,1].hist(dataframe_in['SepalWidthCm'],bins=5)

    axs[1,0].set_title("Petal Length")
    axs[1,0].hist(dataframe_in['PetalLengthCm'],bins=6)

    axs[1,1].set_title("Petal Width")
    axs[1,1].hist(dataframe_in['PetalWidthCm'],bins=6)

    plt.show()

#VarietiesBars(dataframe)


### Visualize box plot 

### Here we visualize a box-and-whisker  to 
# see numerical data distribution
# 1) Choosing y = vertical as specific feature of Iris plant
# 2) Choosing x = horizontal as all species of Iris plant 
def BoxPlotGraph(feature,dataframe_in):
    plt.figure(figsize=(10,7))
    sns.boxplot(x='Species',y=feature,data=dataframe_in)
    plt.show()

# feature = 'SepalLengthCm'
# BoxPlotGraph(feature,dataframe)


### Let's focus on specific feature 
def BoxFeature(feature,dataframe_in):
    plt.figure(figsize=(24,7))
    sns.boxplot(x = feature , data = dataframe_in)
    plt.show()

# BoxFeature('SepalWidthCm',dataframe)



### Heatmap is a tool that will show the correlation between the values of Iris data
# we'll use pearson correlation coefficient to measure linear correlation 
# between features of data set.
def heatmapTool(dataframe_in):
    plt.figure(figsize=(24,7))
    sns.heatmap(dataframe_in.select_dtypes(include=['number']).corr(method='pearson').drop(['Id'],axis=1).drop(['Id'],axis=0),
                annot = True)
    plt.show()

#heatmapTool(dataframe)


def PlottingAccuracy(losses_in, ongoingAcc_in):

    fig,axs = plt.subplots(1,2,figsize=(13,4))

    axs[0].plot(losses.detach() )
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('epoch')
    axs[0].set_title('Losses')

    axs[1].plot(ongoingAcc)
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_title('Accuracy')

    plt.show()



### Now we will try use features of data-set to predict the right category.
# As an output we have the probability about  three types of Iris plant.

###Using method copy() , to keep untouchable the main dataframe
iris = dataframe.copy()


IrisData = torch.tensor( iris[iris.columns[1:5]].values ).float()
#print(IrisData)

### Cannot work with string as a label output . So we have to correspond those labels to numerical form

labels = torch.zeros(len(IrisData ),dtype=torch.long )
labels[iris.Species == 'Iris-setosa'] = 0
labels[iris.Species == 'Iris-versicolor'] = 1
labels[iris.Species == 'Iris-virginica'] = 2

#print(labels)


# model of neural network
ANNiris = nn.Sequential(
    nn.Linear(4,64), 
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,3)
)


# Loss function
lossfun = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(ANNiris.parameters(),lr=.01)

numepochs = 1000

# initialize losses
losses = torch.zeros(numepochs)
ongoingAcc = []

# loop over epochs
for epochi in range(numepochs):
    
    # forward pass
    yHat = ANNiris(IrisData)

    # compute loss
    loss = lossfun(yHat,labels)

    losses[epochi] = loss

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute accuracy
    matches = torch.argmax(yHat,axis=1) == labels
    matchesNumeric = matches.float() 
    accuracyPct = 100*torch.mean(matchesNumeric)
    ongoingAcc.append(accuracyPct)



# final forward pass
predictions = ANNiris(IrisData)

predlabels = torch.argmax(predictions,axis=1)
totalacc = 100*torch.mean((predlabels == labels).float())



# report accuracy
print('Final accuracy: %g%%' %totalacc)


PlottingAccuracy(losses, ongoingAcc)
























