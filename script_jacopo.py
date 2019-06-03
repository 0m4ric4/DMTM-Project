import pandas as pd
import numpy as np
import seaborn as sns

sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def print_info_about_dataset(dt,name):
    print(name)
    print(dt.info())

def check_missing_values(dt):
    print(dt.isnull().sum())

# loading data
events = pd.read_csv("C:\\Users\erica\Desktop\jacopo\progetto dmtm\events_train.csv.gz")

#2)events dataset
#key 2 attribute is useless
events = events.drop('KEY_2', axis=1)
events['START_DATETIME_UTC'] = pd.to_datetime(events['START_DATETIME_UTC'])
events['END_DATETIME_UTC'] = pd.to_datetime(events['START_DATETIME_UTC'])
events['KEY'] = events['KEY'].astype('int64')
check_missing_values(events)  #==> 24 missing values on event detail
#since the percentage of missing values is really small(<0.1%) ==> delete all the rows with mv
events = events.dropna()
events['EVENT_TYPE'] = events['EVENT_TYPE'] + events['EVENT_DETAIL'].map(str)
print(events['EVENT_TYPE'].nunique())
print(events.info())
#events = events.set_index(['KEY'])
#events = events.sort_index()
print(events.head(10))
'''
# Avoiding duplicate columns
events.rename(columns={'KEY': 'KEY_EVENTS', 'KEY_2': 'KEY_2_EVENTS'}, inplace=True)

# The values to be compared.

a = speed_sensors['KM'].values
bh = events['KM_END'].values
bl = events['KM_START'].values

aid = speed_sensors['KEY'].values
bid = events['KEY_EVENTS'].values

at = speed_sensors['DATETIME_UTC'].values
btl = events['START_DATETIME_UTC'].values
bth = events['END_DATETIME_UTC'].values

i, j = np.where(
    (aid[:, None] == bid) & (a[:, None] >= bl) & (a[:, None] <= bh) & (at[:, None] >= btl) & (at[:, None] <= bth))

merged = pd.DataFrame(
    np.column_stack([speed_sensors.values[i], events.values[j]]),
    columns=speed_sensors.columns.append(events.columns)
).append(
    speed_sensors[~np.in1d(np.arange(len(speed_sensors)), np.unique(i))],
    ignore_index=True, sort=False
)

print(merged.head(100))
print(merged.info())
print(merged.shape)

print(merged.count().tail())

#print(missing_values_table(merged))
'''







