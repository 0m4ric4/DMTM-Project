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


########################################################################################################################
#                                                                                                                      #
#                                                   Pre-processing                                                     #
#                                                                                                                      #
########################################################################################################################

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

def print_info_about_dataset(dt,name):
    print(name)
    print(dt.info())

def check_missing_values(dt):
    print(dt.isnull().sum())

def data_exploration(dataset):
    print(dataset.info())
    print(dataset.isnull().sum())
    print(dataset.head(100))


def down_casting_data_types(speeds_chunk):
    # Down-casting the data types of some columns to 32-bit in order to reduce memory usage
    speeds_chunk['KM'] = speeds_chunk['KM'].astype(np.int32)
    speeds_chunk['KEY'] = speeds_chunk['KEY'].astype(np.int32)
    speeds_chunk['SPEED_AVG'] = speeds_chunk['SPEED_AVG'].astype(np.float32)
    speeds_chunk['SPEED_SD'] = speeds_chunk['SPEED_SD'].astype(np.float32)
    speeds_chunk['SPEED_MIN'] = speeds_chunk['SPEED_MIN'].astype(np.float32)
    speeds_chunk['SPEED_MAX'] = speeds_chunk['SPEED_MAX'].astype(np.float32)
    speeds_chunk['N_VEHICLES'] = speeds_chunk['N_VEHICLES'].astype(np.int32)
    return speeds_chunk


def merge_speed_and_sensors(speed_chunk, sensors):
    # Merging speeds.train.csv.gz with sensors.csv.gz
    speeds_sensors = pd.merge(speed_chunk, sensors, on=['KEY', 'KM']).drop_duplicates().reset_index(
        drop=True)
    return speeds_sensors

def merge_SpeedSensors_and_events(speeds_chunk,events):
    # Merging speeds_sensors with events_train.csv.gz

    sensor_km = speeds_sensors.KM.values

    event_km_start = events.KM_START.values
    event_km_end = events.KM_END.values

    speed_sensors_key = speeds_sensors.KEY.values
    events_key = events.KEY_EVENTS.values

    speeds_sensors_time = speeds_sensors.DATETIME_UTC.values
    events_time_start = events.START_DATETIME_UTC.values
    events_time_end = events.END_DATETIME_UTC.values

    # The conditions for the merge
    i, j = np.where(
        (speed_sensors_key[:, None] == events_key) & (
                ((sensor_km[:, None] >= event_km_start) & (sensor_km[:, None] <= event_km_end)) | (
                (sensor_km[:, None] <= event_km_start) & (sensor_km[:, None] >= event_km_end))) & (
                speeds_sensors_time[:, None] >= events_time_start) & (speeds_sensors_time[:, None] <= events_time_end))

    # Performing the merge based on the above conditions
    speeds_sensors_events = pd.DataFrame(
        np.column_stack([speeds_sensors.values[i], events.values[j]]),
        columns=speeds_sensors.columns.append(events.columns)
    ).append(
        speeds_sensors[~np.in1d(np.arange(len(speeds_sensors)), np.unique(i))],
        ignore_index=True, sort=False
    )
    return speeds_sensors_events


########################################################################################################################
#                                                                                                                      #
# Merging speeds_train.csv.gz with sensors.csv.gz and events_train.csv.gz                                              #
#                                                                                                                      #
########################################################################################################################

SPEED_TRAIN_PATH = "C:\\Users\erica\Desktop\jacopo\progetto dmtm\speeds_train.csv.gz"
SENSORS_PATH = "C:\\Users\erica\Desktop\jacopo\progetto dmtm\sensors.csv.gz"
EVENT_TRAIN_PATH = "C:\\Users\erica\Desktop\jacopo\progetto dmtm\events_train.csv.gz"
WEATHER_TRAIN_PATH = "C:\\Users\erica\Desktop\jacopo\progetto dmtm\weather_train.csv.gz"
DISTANCES_PATH ="C:\\Users\erica\Desktop\jacopo\progetto dmtm\distances.csv.gz"
PATH_TO_SAVE_THE_FINAL_DATASET = "C:\\Users\erica\Desktop\jacopo\progetto dmtm\dataset_final.csv"


speed_chunks = pd.read_csv(SPEED_TRAIN_PATH,
                           chunksize=150000)  # Specifying chunk size for the speed dataset to be 150,000 rows
sensors = pd.read_csv(SENSORS_PATH)
events = pd.read_csv(EVENT_TRAIN_PATH)

#events preprocessing
#cast dates
events['START_DATETIME_UTC'] = pd.to_datetime(events['START_DATETIME_UTC'])
events['END_DATETIME_UTC'] = pd.to_datetime(events['END_DATETIME_UTC'])
events['EVENT_DETAIL'] = events['EVENT_DETAIL'].astype(np.int32)
events['KEY'] = events['KEY'].astype(np.int32)
events['KM_END'] = events['KM_END'].astype(np.int32)
events['KM_START'] = events['KM_START'].astype(np.int32)
check_missing_values(events)
#==> 24 missing values on event detail
#since the percentage of missing values is really small(<0.1%) ==> delete all the rows with mv
events = events.dropna()

# Avoiding duplicate columns
events.rename(columns={'KEY': 'KEY_EVENTS', 'KEY_2': 'KEY_2_EVENTS'}, inplace=True)

chunk_list = []  # append each chunk df here

for chunk in speed_chunks:

    chunk = down_casting_data_types(chunk)
    chunk['DATETIME_UTC'] = pd.to_datetime(chunk['DATETIME_UTC'])
    print(chunk.info())

    # perform merging
    speeds_sensors = merge_speed_and_sensors(chunk,sensors)
    # perform merging
    chunk_processed = merge_SpeedSensors_and_events(speeds_sensors,events)

    # Once the data processing is done, append the chunk to list
    chunk_list.append(chunk_processed)

# concat the list into dataframe
dataset = pd.concat(chunk_list)


# Eliminating duplicate timestamps by appending concurrent events to a new column
# Maximum concurrent events = 4
# New columns: Event1, Event2, Event3, Event4

dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])
# dataset['DATETIME_UTC'] = dataset['DATETIME_UTC'] + pd.to_timedelta(dataset.groupby('DATETIME_UTC').cumcount(), unit='ms')
dataset.set_index('DATETIME_UTC')

dataset.drop('START_DATETIME_UTC', axis=1, inplace=True)
dataset.drop('END_DATETIME_UTC', axis=1, inplace=True)
dataset.drop('EVENT_TYPE', axis=1, inplace=True)
dataset.drop('KM_START', axis=1, inplace=True)
dataset.drop('KM_END', axis=1, inplace=True)
dataset.drop('KEY_EVENTS', axis=1, inplace=True)
dataset.drop('KEY_2_EVENTS', axis=1, inplace=True)

print(dataset.info())
print(dataset.head(20))
print('..')

g = dataset.groupby(
    ["KEY", "DATETIME_UTC", "KM", "SPEED_AVG", "SPEED_SD", "SPEED_MIN", "SPEED_MAX", "N_VEHICLES", "KEY_2",
     "EMERGENCY_LANE",
     "LANES", "ROAD_TYPE"]).cumcount().add(1)
dataset = dataset.set_index(
    ["KEY", "DATETIME_UTC", "KM", "SPEED_AVG", "SPEED_SD", "SPEED_MIN", "SPEED_MAX", "N_VEHICLES", "KEY_2",
     "EMERGENCY_LANE",
     "LANES", "ROAD_TYPE", g]).unstack(fill_value=-1).sort_index(axis=1, level=1)
dataset.columns = ["{}{}".format(a, b) for a, b in dataset.columns]

dataset['EVENT_DETAIL1'].fillna(-1, inplace=True)

dataset = dataset.reset_index()

dataset.sort_values(by=["KEY", "KM", "DATETIME_UTC"], inplace=True)

# Merging the current dataset with distances.csv.gz and weather_train.csv.gz


# reading distances.csv.gz with '|' as the delimiter, with 2 columns (ColA and ColB)
distances = pd.read_csv(DISTANCES_PATH, delimiter='|', names=['ColA', 'ColB'])
distances.dropna(inplace=True)

# Creating an empty data-frame to store the nearest weather station with attributes (KEY_2 and ID)

distances_processed = pd.DataFrame(columns=['KEY_2', 'ID'])

# Iterating over distances.cv.gz to find the nearest weather station (ID) to each sensor (KEY_2)

for index, row in distances.iterrows():

    # Extracting KEY_2
    key = row["ColA"].split(',')[0] + "_" + row["ColA"].split(',')[1]

    # Creating a tuple storing the weather station ID, distances
    # Ex: (STATION_29,10,STATION_37,19,STATION_36,40,STATION_30,64,STATION_33,94)
    dist_tuple = tuple(row["ColB"].split(','))

    # Setting the minimum distance to be that of first station in the tuple
    min_index = 1
    min = float(dist_tuple[min_index])
    print(index)

    # Iterating over the odd numbers in the tuple to find the nearest station starting index 3 (since index 1 is already the minimum till now)
    for i in range(3, len(dist_tuple), 2):
        if float(dist_tuple[i]) < min:
            min = float(dist_tuple[i])
            min_index = i
        # print(i, dist_tuple[i])
    distances_processed.loc[index] = [key] + [dist_tuple[min_index - 1]]

# Sample of the output of distances_processed
# KEY_2	ID
# 278_662	STATION_29
# 278_663	STATION_29
# 278_664	STATION_29
# 278_665	STATION_29
# 278_666	STATION_29
# 278_667	STATION_29


# Reading weather_train.csv.gz
weather = pd.read_csv(WEATHER_TRAIN_PATH)

# Converting DATETIME_UTC to datetime dtype
dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])
weather['DATETIME_UTC'] = pd.to_datetime(weather['DATETIME_UTC'])

# Merging dataset with distances_processed
df = pd.merge(dataset, distances_processed, on='KEY_2')

# Merging on DATETIME_UTC requires both data-frames to be sorted on DATETIME_UTC
df.sort_values(by='DATETIME_UTC', inplace=True)
weather.sort_values(by='DATETIME_UTC', inplace=True)

# Merging on the nearest time difference after combining by the weather station ID
dataset_final = pd.merge_asof(df, weather, on='DATETIME_UTC', by='ID', direction='nearest')
dataset_final.sort_values(by=["KEY", "KM", "DATETIME_UTC"], inplace=True)

# Exporting to dataset_final.csv
dataset_final.to_csv(PATH_TO_SAVE_THE_FINAL_DATASET, encoding='utf-8', index=False)
