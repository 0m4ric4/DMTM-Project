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

def data_preparation(dataset):
    #convert the data into inputs and targets
    inputs = []
    targets = []

#loading data
full_dataset = pd.read_csv('C:\\Users\erica\Desktop\jacopo\progetto dmtm\\full_dataset.csv')

#delete useless columns in term of prediction
full_dataset.drop('KEY',axis=1,inplace=True)
full_dataset.drop('KM',axis=1,inplace=True)
full_dataset.drop('SPEED_SD',axis=1,inplace=True)
full_dataset.drop('SPEED_MIN',axis=1,inplace=True)
full_dataset.drop('SPEED_MAX',axis=1,inplace=True)
full_dataset.drop('ID',axis=1,inplace=True)
full_dataset.drop('MIN_TEMPERATURE',axis=1,inplace=True)
full_dataset.drop('MAX_TEMPERATURE',axis=1,inplace=True)
full_dataset['WEATHER'] = full_dataset['WEATHER'].astype('category').cat.codes

data_exploration(full_dataset)



