import pandas as pd

pd.options.display.max_columns = 100
import numpy as np


def data_exploration(dataset):
    print(dataset.info())
    print(dataset.isnull().sum())
    print(dataset.head(100))


def add_columns_to_dataframe(dataframe, list_of_events):
    for event in list_of_events:
        dataframe[event] = np.int8(0)
    return dataframe


def add_ones(dataframe):
    for index, row in dataframe.iterrows():
        print(index)
        for i in range(1, 3):
            if (row['EVENT_TYPE' + str(i)] != 'NO_EVENT'):
                event_type = row['EVENT_TYPE' + str(i)]
                dataframe.at[index, event_type] = np.int(1)
    return dataframe


dataset = pd.read_csv('test_5_june_2.csv')

# compute a list of all the possible events occuring
#list_of_events = dataset[['EVENT_TYPE1', 'EVENT_TYPE2', 'EVENT_TYPE3', 'EVENT_TYPE4']].stack().unique()
list_of_events = ['NO_EVENT', 'Veicolo_in_avaria', 'Ostacolo_in_carreggiata', 'Regimazione_delle_acque', 'Manutenzione_opere_in_verde', 'Meteo', 'Gestione_viabilita', 'extended_accident', 'Segnaletica_orizzontale', 'Allarme', 'Segnaletica_verticale', 'Pavimentazione', 'Barriere', 'Opera_arte', 'Calamita_naturale']

print(list_of_events)
print(len(list_of_events))

# delete no event since we don't want a column with no event or nan values
list_of_events.remove('NO_EVENT')
for string in list_of_events:
    if "nan" in string:
        list_of_events.remove(string)

# add columns to the dataset, one for each event (for now all with zeros)

dataset = add_columns_to_dataframe(dataset, list_of_events)

dataset = add_ones(dataset)

print(dataset.info())

dataset.to_csv('test_final.csv', encoding='utf-8', index=False)
