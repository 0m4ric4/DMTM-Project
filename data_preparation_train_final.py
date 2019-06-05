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

def adding_ones(dataframe):
    for index,row in dataframe.iterrows():
        print(index)
        for i in range(1,5):
            if(row['EVENT_TYPE' + str(i)] != "NO_EVENT"):
                event_type = row['EVENT_TYPE' + str(i)]
                dataframe.at[index,event_type] = np.int8(1)
    return dataframe


dataframe_dict= pd.read_csv('dict_to_cast_events_when_loading.csv',delimiter=';')
dict = {}
for index,row in dataframe_dict.iterrows():
    dict[row['EVENTO']] = row['TIPO']
print(dict)

dataframe= pd.read_csv('C:\\Users\erica\Desktop\jacopo\progetto dmtm\\final_train.csv',dtype=dict)
dataframe = adding_ones(dataframe)
data_exploration(dataframe)