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

dataset = pd.read_csv(DATAFRAME_PATH,dtype=dict)
data_exploration(dataset)

