import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataframe = pd.read_csv("Zomato-data-.csv")
print(dataframe.head())

### Here its significant to mention .copy() function
# And i have to explain usage of it
rateColumn = dataframe['rate'].copy()
print("### Rate column")
print(rateColumn[2].split(sep='/'))


for idx,value in enumerate(rateColumn):
    clean = value.split(sep='/')[0]
    rateColumn[idx] = float(clean)

### Column values are object. Below we converting objct type to float64
rateColumn = rateColumn.astype(float)

dataframe['rate'] = rateColumn.copy()

# print(dataframe)
# print(dataframe.info())

### Have our Dataframe got any null or gap value? Have to check
# print( dataframe.isnull().sum() )

### countplot function o seaborn library
# sns.countplot(x = dataframe['listed_in(type)'])
# plt.ylabel("Count of restaurant")
# plt.xlabel("Type of restaurant")
# plt.show()


### i have to plot count of votes for each category
# here we using pandas function groupby which is inspired from SQL

grouped_data = dataframe.groupby('listed_in(type)')['votes'].sum()
# print(grouped_data)
# result = pd.DataFrame({'votes':grouped_data})
# plt.plot(result,color='black')
# plt.xlabel('type of restaurant')
# plt.ylabel('Votes which count for each restaurant')
# plt.show()

### Trying to make clear the relationship between order [online_order] and restaurant type[listed_in(type)]

pivot_table = dataframe.pivot_table(index = 'listed_in(type)',columns = 'online_order',aggfunc='size',
                                    fill_value=0)
sns.heatmap(pivot_table , annot=True ,cmap = 'YlGnBu' , fmt = 'd')
plt.title('Heatmap')
plt.xlabel('Online Order')
plt.ylabel('Listed In (Type)')
plt.show()

















































































