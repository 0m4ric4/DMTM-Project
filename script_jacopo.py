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

def set_km_start(event):
    if (event['KM_START'] >= event['KM_END']):
        kmstart = event['KM_START']
        kmend = event['KM_END']
    else:
        kmstart = event['KM_END']
        kmend = event['KM_START']
    return zip(kmstart, kmend)


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns

# loading data
speed = pd.read_csv("C:\\Users\erica\Desktop\jacopo\progetto dmtm\speeds_train.csv.gz")
weather = pd.read_csv("C:\\Users\erica\Desktop\jacopo\progetto dmtm\weather_train.csv.gz")
events = pd.read_csv("C:\\Users\erica\Desktop\jacopo\progetto dmtm\events_train.csv.gz")
sensors = pd.read_csv("C:\\Users\erica\Desktop\jacopo\progetto dmtm\sensors.csv.gz")


####### Info about the datasets before merge #######
print_info_about_dataset(speed,'SPEED')
print_info_about_dataset(events,'EVENTS')
print_info_about_dataset(sensors,'SENSORS')
print_info_about_dataset(weather,'WEATHER')

####### Pre-Processing ########
#1)speed dataset
speed = speed.drop('KEY_2', axis=1)
speed['DATETIME_UTC'] = pd.to_datetime(speed['DATETIME_UTC'])
check_missing_values(speed)
speed = speed.set_index(['KEY','KM','DATETIME_UTC']) #primary key as index
speed = speed.sort_index()
print (speed.head(10))

#2)events dataset
#key 2 attribute is useless
events = events.drop('KEY_2', axis=1)
#event type is useless since we use event detail that is the same but remapped to int values
events = events.drop('EVENT_TYPE', axis=1)
events['START_DATETIME_UTC'] = pd.to_datetime(events['START_DATETIME_UTC'])
events['END_DATETIME_UTC'] = pd.to_datetime(events['START_DATETIME_UTC'])
events['KEY'] = events['KEY'].astype('int64')
check_missing_values(events)  #==> 24 missing values on event detail
#since the percentage of missing values is really small(<0.1%) ==> delete all the rows with mv
events = events.dropna()
print(events.info())
events = events.set_index(['KEY'])
events = events.sort_index()
print(events.head(10))

#3)sensors dataset
check_missing_values(sensors)
sensors = sensors.set_index(['KEY','KM'])
sensors = sensors.sort_index()
print(sensors.head(10))

#4)weather dataset
check_missing_values(weather)
weather['DATETIME_UTC'] = pd.to_datetime(weather['DATETIME_UTC'])


# merging speed and sensors datasets
speed_sensors = speed.join(sensors).drop_duplicates()
print(speed_sensors.head(5))
print(speed_sensors.info())

#merging speed_sensors with events dataset
for event in events.iterrows():
    [km_start, km_end] = set_km_start(event)
    #to finish

'''
# Avoiding duplicate columns
events.rename(columns={'KEY': 'KEY_EVENTS', 'KEY_2': 'KEY_2_EVENTS'}, inplace=True)

# The values to be compared.

a = speed_sensors.KM.values
bh = events.KM_END.values
bl = events.KM_START.values

aid = speed_sensors.KEY.values
bid = events.KEY_EVENTS.values

at = speed_sensors.DATETIME_UTC.values
btl = events.START_DATETIME_UTC.values
bth = events.END_DATETIME_UTC.values

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

print(missing_values_table(merged))
'''







