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


def chunk_preprocessing(speed_chunk):

    speed_chunk['KM'] = speed_chunk['KM'].astype(np.int32)
    speed_chunk['KEY'] = speed_chunk['KEY'].astype(np.int32)
    speed_chunk['SPEED_AVG'] = speed_chunk['SPEED_AVG'].astype(np.float32)
    speed_chunk['SPEED_SD'] = speed_chunk['SPEED_SD'].astype(np.float32)
    speed_chunk['SPEED_MIN'] = speed_chunk['SPEED_MIN'].astype(np.float32)
    speed_chunk['SPEED_MAX'] = speed_chunk['SPEED_MAX'].astype(np.float32)
    speed_chunk['N_VEHICLES'] = speed_chunk['N_VEHICLES'].astype(np.int32)
    speed_chunk['DATETIME_UTC'] = pd.to_datetime(speed_chunk['DATETIME_UTC'])

    print(speed_chunk.info())

    speed_sensors = pd.merge(speed_chunk, sensors, on=['KEY', 'KM']).drop_duplicates().reset_index(
        drop=True)

    print(speed_sensors.info())
    print(speed_sensors.head(20))

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

    return merged


#Specifying chunk size for the speed dataset to be 300,000 rows
speed_chunks = pd.read_csv('speeds_train.csv.gz', chunksize=300000)
sensors = pd.read_csv('sensors.csv.gz')
events = pd.read_csv('events_train.csv.gz')


events['START_DATETIME_UTC'] = pd.to_datetime(events['START_DATETIME_UTC'])
events['END_DATETIME_UTC'] = pd.to_datetime(events['END_DATETIME_UTC'])
# events['EVENT_DETAIL'] = events['EVENT_DETAIL'].astype(np.int32)
events['KEY'] = events['KEY'].astype(np.int32)
events['KM_END'] = events['KM_END'].astype(np.int32)
events['KM_START'] = events['KM_START'].astype(np.int32)

# Avoiding duplicate columns
events.rename(columns={'KEY': 'KEY_EVENTS', 'KEY_2': 'KEY_2_EVENTS'}, inplace=True)

####### Info about the datasets before merge #######


print(sensors.info())
print(events.info())

####### Pre-Processing ########

chunk_list = []  # append each chunk df here

# Each chunk is in df format

i = 1
for chunk in speed_chunks:

    print("Processing chunk # "+str(i))
    i = i+1
    # perform data filtering
    chunk_filter = chunk_preprocessing(chunk)

    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk_filter)

# concat the list into dataframe
df_concat = pd.concat(chunk_list)


# Save concatenated dataframe to csv
df_concat.to_csv('dataset.csv', encoding='utf-8', index=False)
