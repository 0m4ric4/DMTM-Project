import pandas as pd

pd.options.display.max_columns = 100

import numpy as np

# visualization library
import seaborn as sns

sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})

# import matplotlib and allow it to plot inline
import matplotlib.pyplot as plt
# matplotlib inline

# seaborn can generate several warnings, we ignore them
import warnings

warnings.filterwarnings("ignore")


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


speed = pd.read_csv('speeds_train.csv.gz')
sensors = pd.read_csv('sensors.csv.gz')
events = pd.read_csv('events_train.csv.gz')

speed['KM'] = speed['KM'].astype(np.int32)
speed['KEY'] = speed['KEY'].astype(np.int32)
speed['SPEED_AVG'] = speed['SPEED_AVG'].astype(np.float32)
speed['SPEED_SD'] = speed['SPEED_SD'].astype(np.float32)
speed['SPEED_MIN'] = speed['SPEED_MIN'].astype(np.float32)
speed['SPEED_MAX'] = speed['SPEED_MAX'].astype(np.float32)
speed['N_VEHICLES'] = speed['N_VEHICLES'].astype(np.int32)
speed['DATETIME_UTC'] = pd.to_datetime(speed['DATETIME_UTC'])

events['START_DATETIME_UTC'] = pd.to_datetime(events['START_DATETIME_UTC'])
events['END_DATETIME_UTC'] = pd.to_datetime(events['END_DATETIME_UTC'])
# events['EVENT_DETAIL'] = events['EVENT_DETAIL'].astype(np.int32)
events['KEY'] = events['KEY'].astype(np.int32)
events['KM_END'] = events['KM_END'].astype(np.int32)
events['KM_START'] = events['KM_START'].astype(np.int32)

####### Info about the datasets before merge #######


print(speed.info())
print(sensors.info())
print(events.info())

####### Pre-Processing ########


### Taking the first 50000 samples only ###
speed_sensors = pd.merge(speed, sensors, on=['KEY', 'KM']).drop_duplicates().head(50000).reset_index(drop=True)

print(speed_sensors.info())
print(speed_sensors.head(20))

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
print(merged.info())

print(merged.count().tail())

print(missing_values_table(merged))
