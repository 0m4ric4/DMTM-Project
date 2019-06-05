import pandas as pd
pd.options.display.max_columns = 100
import numpy as np
# visualization library
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})
# seaborn can generate several warnings, we ignore them
import warnings
warnings.filterwarnings("ignore")

def data_exploration(dataset):
    print(dataset.info())
    print(dataset.isnull().sum())
    print(dataset.head(100))

def to_supervised(dataset):
    inputs = np.array([dataset.shape[0] - 4,dataset.shape[1]])
    targets = [dataset.shape[0],4]
    for index,row in dataset.iterrows():
        if(index <= inputs.shape[0]):
            inputs[index] = row

DATAFRAME_PATH = 'C:\\Users\erica\Desktop\jacopo\progetto dmtm\\full_dataset_after_preprocess2.csv'
dataframe_dict= pd.read_csv('dictionary.csv')
dict = {}
for index,row in dataframe_dict.iterrows():
    dict[row['EVENTO']] = row['TIPO']
print(dict)

def adding_ones(dataframe):
    for index,row in dataframe.iterrows():
        print(index)
        for i in range(1,5):
            if(row['EVENT_TYPE'] + str(i)) != "NO_EVENT":
                event_type = row['EVENT_TYPE'] + str(i)
                row[event_type] = np.int8(1)
    return dataframe

dataset = pd.read_csv(DATAFRAME_PATH,dtype=dict)
dataset.drop('Unnamed: 0', axis=1,inplace=True)
data_exploration(dataset)
i = 0
for key2, df_key2 in dataset.groupby('KEY_2'):
    print(str(i) + ") " + key2)
    df_key2.to_csv('C:\\Users\erica\Desktop\jacopo\progetto dmtm\\datasets\dataset_sensor_' + str(key2) + ".csv")
    i +=1

'''
data_exploration(dataset)
dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])

#one-hot-encoding the weather 
dataframe_weather = pd.get_dummies(dataset['WEATHER'],prefix='WEATHER',dtype='int8')
dataset.append(dataframe_weather)
dataset.drop('WEATHER', axis=1,inplace=True)
data_exploration(dataset)
'''
