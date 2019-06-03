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

def find_unique_events(event_columns):
    already_listed = []
    index = 0
    for i in range(1,5):
        for event in event_columns['EVENT_TYPE' + str(i)]:
            print (index)
            if(event not in already_listed):
                already_listed.append(event)
            index += 1
    return already_listed

def add_columns_to_dataframe(dataframe,list_of_events):
    for event in list_of_events:
        dataframe[event] = 0
        dataframe[event] = pd.to_numeric(dataframe[event],downcast='integer')
    return dataframe


dataset = pd.read_csv('C:\\Users\erica\Desktop\jacopo\progetto dmtm\\full_dataset_after_preprocess.csv')
dataset.drop('Unnamed: 0', axis=1,inplace=True)

#compute a list of all the possible events occuring
list_of_events= find_unique_events(dataset[['EVENT_TYPE1','EVENT_TYPE2','EVENT_TYPE3','EVENT_TYPE4']])
print(list_of_events)
print(len(list_of_events))

#delete no event since we don't want a column with no event or nan values
list_of_events.remove('NO_EVENT')
for string in list_of_events:
    if "nan" in string:
        list_of_events.remove(string)

#add columns to the dataset, one for each event (for now all with zeros)
dataframe = add_columns_to_dataframe(dataset, list_of_events)
data_exploration(dataframe)
dataframe.to_csv('C:\\Users\erica\Desktop\jacopo\progetto dmtm\\full_dataset_after_preprocess2.csv')

