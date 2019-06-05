import numpy as np

# visualization library
import seaborn as sns
import pandas as pd

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


def chunk_preprocessing(speeds_chunk):
    # Down-casting the data types of some columns to 32-bit in order to reduce memory usage
    speeds_chunk['KM'] = speeds_chunk['KM'].astype(np.int32)
    speeds_chunk['KEY'] = speeds_chunk['KEY'].astype(np.int32)
    speeds_chunk['SPEED_AVG'] = speeds_chunk['SPEED_AVG'].astype(np.float32)
    speeds_chunk['SPEED_SD'] = speeds_chunk['SPEED_SD'].astype(np.float32)
    speeds_chunk['SPEED_MIN'] = speeds_chunk['SPEED_MIN'].astype(np.float32)
    speeds_chunk['SPEED_MAX'] = speeds_chunk['SPEED_MAX'].astype(np.float32)
    speeds_chunk['N_VEHICLES'] = speeds_chunk['N_VEHICLES'].astype(np.int32)
    speeds_chunk['DATETIME_UTC'] = pd.to_datetime(speeds_chunk['DATETIME_UTC'])

    print(speeds_chunk.info())

    # Merging speeds.test.csv.gz with sensors.csv.gz
    speeds_sensors = pd.merge(speeds_chunk, sensors, on=['KEY', 'KM']).drop_duplicates().reset_index(
        drop=True)

    # Merging speeds_sensors with events_test.csv.gz

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
# Merging speeds_test.csv.gz with sensors.csv.gz and events_test.csv.gz                                              #
#                                                                                                                      #
########################################################################################################################


speed_chunks = pd.read_csv('speeds_2019.csv',
                           chunksize=300000)  # Specifying chunk size for the speed dataset to be 300,000 rows
sensors = pd.read_csv('sensors.csv.gz')
events = pd.read_csv('events_2019.csv')

events['START_DATETIME_UTC'] = pd.to_datetime(events['START_DATETIME_UTC'])
events['END_DATETIME_UTC'] = pd.to_datetime(events['END_DATETIME_UTC'])
# events['EVENT_DETAIL'] = events['EVENT_DETAIL'].astype(np.int32)
events['KEY'] = events['KEY'].astype(np.int32)
events['KM_END'] = events['KM_END'].astype(np.int32)
events['KM_START'] = events['KM_START'].astype(np.int32)

# Avoiding duplicate columns
events.rename(columns={'KEY': 'KEY_EVENTS', 'KEY_2': 'KEY_2_EVENTS'}, inplace=True)

chunk_list = []  # append each chunk df here

# Each chunk is in df format

i = 1
for chunk in speed_chunks:
    chunk.dropna(inplace=True)
    print("Processing chunk # " + str(i))
    i = i + 1
    # perform data pre-processing
    chunk_processed = chunk_preprocessing(chunk)

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

dataset['EVENT_TYPE'] = dataset['EVENT_TYPE'] + '_' + dataset['EVENT_DETAIL'].map(str)

dataset.drop('EVENT_DETAIL', axis=1, inplace=True)
dataset.drop('START_DATETIME_UTC', axis=1, inplace=True)
dataset.drop('END_DATETIME_UTC', axis=1, inplace=True)
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
     "LANES", "ROAD_TYPE", g]).unstack(fill_value='NO_EVENT').sort_index(axis=1, level=1)
dataset.columns = ["{}{}".format(a, b) for a, b in dataset.columns]

dataset['EVENT_TYPE1'].fillna('NO_EVENT', inplace=True)

dataset = dataset.reset_index()

dataset.sort_values(by=["KEY", "KM", "DATETIME_UTC"], inplace=True)

# Merging the current dataset with distances.csv.gz and weather_test.csv.gz


# reading distances.csv.gz with '|' as the delimiter, with 2 columns (ColA and ColB)
distances = pd.read_csv('distances.csv.gz', delimiter='|', names=['ColA', 'ColB'])
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


# Reading weather_test.csv.gz
weather = pd.read_csv('weather_2019.csv')

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
dataset_final.to_csv('test_2019.csv', encoding='utf-8', index=False)