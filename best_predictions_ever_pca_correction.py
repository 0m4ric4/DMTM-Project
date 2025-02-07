from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

import pandas as pd

'''
mean absolute error:
5.36000286759674
mean squared error:
62.23639729185435
'''
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


def to_supervised(data, dropNa=True, lag=1):
    df = pd.DataFrame(data)
    column = []
    column.append(df)
    for i in range(1, lag + 1):
        column.append(df.shift(-i))
    df = pd.concat(column, axis=1)
    df.dropna(inplace=True)
    features = data.shape[1]
    df = df.values
    supervised_data = df[:, :features * lag]
    supervised_data = np.column_stack([supervised_data, df[:, features * lag]])
    return supervised_data


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):

        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


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


dataframe_dict = pd.read_csv('dict_to_cast_events_when_loading.csv', delimiter=";")
dictionary = {}
for index, row in dataframe_dict.iterrows():
    dictionary[row['EVENTO']] = row['TIPO']
print(dictionary)

train = pd.read_csv('train_final.csv', dtype=dictionary)
test = pd.read_csv('test_2019_final.csv', dtype=dictionary)

print(train.info())
print(test.info())

df_temp_train = train[['WEATHER']]
df_temp_test = test[['WEATHER']]

df_temp_train['train'] = 1
df_temp_test['train'] = 0

df_combined = pd.concat([df_temp_train, df_temp_test])

df_combined_encoded = pd.get_dummies(df_combined['WEATHER'])

df_combined = pd.concat([df_combined, df_combined_encoded], axis=1)

train_encoded = df_combined[df_combined["train"] == 1]
test_encoded = df_combined[df_combined["train"] == 0]

train_encoded.drop(['train'], axis=1, inplace=True)
train_encoded.drop(['WEATHER'], axis=1, inplace=True)
test_encoded.drop(['train'], axis=1, inplace=True)
test_encoded.drop(['WEATHER'], axis=1, inplace=True)

print(train_encoded.info())
print(test_encoded.info())

print(test_encoded)

train.drop('WEATHER', axis=1, inplace=True)
train.drop('TEMPERATURE', axis=1, inplace=True)
train.drop('MIN_TEMPERATURE', axis=1, inplace=True)
train.drop('MAX_TEMPERATURE', axis=1, inplace=True)
train.drop('ID', axis=1, inplace=True)
train.drop('EVENT_TYPE1', axis=1, inplace=True)
train.drop('EVENT_TYPE2', axis=1, inplace=True)
train.drop('EVENT_TYPE3', axis=1, inplace=True)
train.drop('EVENT_TYPE4', axis=1, inplace=True)
train.drop('KEY', axis=1, inplace=True)
train.drop('KM', axis=1, inplace=True)

test.drop('WEATHER', axis=1, inplace=True)
test.drop('TEMPERATURE', axis=1, inplace=True)
test.drop('MIN_TEMPERATURE', axis=1, inplace=True)
test.drop('MAX_TEMPERATURE', axis=1, inplace=True)
test.drop('ID', axis=1, inplace=True)
test.drop('EVENT_TYPE1', axis=1, inplace=True)
test.drop('EVENT_TYPE2', axis=1, inplace=True)
test.drop('KEY', axis=1, inplace=True)
test.drop('KM', axis=1, inplace=True)

train = pd.concat([train, train_encoded], axis=1)
test = pd.concat([test, test_encoded], axis=1)

print(train.info())
print(train.head(10))

print(test.info())
print(test.head(10))

train_dfs = dict(tuple(train.groupby('KEY_2')))
test_dfs = dict(tuple(test.groupby('KEY_2')))

result_df = pd.DataFrame(columns=['KEY', 'KM', 'DATETIME_UTC', 'PREDICTION_STEP', 'SPEED_AVG'])

train_keys = sorted(train_dfs)
test_keys = sorted(test_dfs)

# We should only train keys that are present in both train and set sets
to_be_trained_keys = list(set(train_keys) & set(test_keys))
# sorts and removes duplicates
to_be_trained_keys = list(OrderedDict.fromkeys(to_be_trained_keys))
to_be_trained_keys = sorted(to_be_trained_keys)

to_be_trained_keys = to_be_trained_keys[320:]

print(to_be_trained_keys)
print("Number of keys to be trained:" + str(len(to_be_trained_keys)))

trained_keys = []

index_of_keys = 1

for key in to_be_trained_keys:

    n_samples_train = len(train_dfs[key].index)
    n_samples_test = len(test_dfs[key].index)

    if (n_samples_train >= 5 and n_samples_test >= 5):

        print("training sensor: " + str(index_of_keys) + " " + str(key))

        trained_keys.append(key)
        train_df = train_dfs[key]

        train_df.set_index('DATETIME_UTC', inplace=True)
        train_df.drop('KEY_2', axis=1, inplace=True)

        test_df = test_dfs[key]

        test_df.set_index('DATETIME_UTC', inplace=True)
        test_df.drop('KEY_2', axis=1, inplace=True)

        # print(train_df.info())
        # print(train_df.head(5))

        values_train = train_df.values

        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(values_train)

        # print(scaled_train.head())

        # test_df.set_index('DATETIME_UTC', inplace=True)

        # print(test_df.info())
        # print(test_df.head(5))

        values_test = test_df.values

        scaler = StandardScaler()
        scaled_test = scaler.fit_transform(values_test)

        if (n_samples_train) > 20 and (n_samples_test > 20):
            n_components_for_pca = 20
        else:
            n_components_for_pca = min(n_samples_train, n_samples_test)

        pca = PCA(n_components_for_pca)

        scaled_train = pca.fit_transform(scaled_train)
        scaled_test = pca.transform(scaled_test)

        explained_variance = pca.explained_variance_ratio_

        timeSteps = 1
        ahead = 4

        supervised_train = series_to_supervised(scaled_train, n_in=timeSteps, n_out=ahead)
        supervised_test = series_to_supervised(scaled_test, n_in=timeSteps, n_out=ahead)

        features_train = len(scaled_train[key].index)
        features_test = len(scaled_test[key].index)

        supervised_train.drop(supervised_train.columns[range(features_train + 1, features_train + 20)], axis=1,
                              inplace=True)
        supervised_train.drop(supervised_train.columns[range(features_train + 2, features_train + 21)], axis=1,
                              inplace=True)
        supervised_train.drop(supervised_train.columns[range(features_train + 3, features_train + 22)], axis=1,
                              inplace=True)
        supervised_train.drop(supervised_train.columns[range(features_train + 4, features_train + 23)], axis=1,
                              inplace=True)

        supervised_test.drop(supervised_test.columns[range(features_test + 1, features_test + 20)], axis=1,
                             inplace=True)
        supervised_test.drop(supervised_test.columns[range(features_test + 2, features_test + 21)], axis=1,
                             inplace=True)
        supervised_test.drop(supervised_test.columns[range(features_test + 3, features_test + 22)], axis=1,
                             inplace=True)
        supervised_test.drop(supervised_test.columns[range(features_test + 4, features_test + 23)], axis=1,
                             inplace=True)

        supervised_train = supervised_train.values
        supervised_test = supervised_test.values

        X_train = supervised_train[:, :features_train * timeSteps]
        y_train = supervised_train[:, features_train * timeSteps:]

        X_test = supervised_test[:, :features_test * timeSteps]
        y_test = supervised_test[:, features_test * timeSteps:]

        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        X_train = X_train.reshape(X_train.shape[0], timeSteps, features_train)
        X_test = X_test.reshape(X_test.shape[0], timeSteps, features_test)

        # print(X_train.shape, X_test.shape)

        NUM_NEURONS_FirstLayer = 80
        NUM_NEURONS_SecondLayer = 50

        EPOCHS = 20

        # Build the model
        model = Sequential()
        model.add(LSTM(NUM_NEURONS_FirstLayer, input_shape=(timeSteps, X_train.shape[2]), return_sequences=True))
        model.add(LSTM(NUM_NEURONS_SecondLayer, input_shape=(NUM_NEURONS_FirstLayer, 1)))

        model.add(Dense(ahead))
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='mae', optimizer=sgd)

        history = model.fit(X_train, y_train, epochs=EPOCHS, shuffle=True, batch_size=32,
                            verbose=2)

        y_pred = model.predict(X_test)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2] * X_test.shape[1])

        y_true = []
        y_predicted = []
        for i in range(0, ahead):
            y_pred_i = y_pred[:, i]
            y_pred_i = y_pred_i.reshape(y_test.shape[0], 1)

            inv_new = np.concatenate((y_pred_i, X_test[:, -19:]), axis=1)
            inv_new = scaler.inverse_transform(pca.inverse_transform(inv_new))
            final_pred = inv_new[:, 0]
            y_predicted.append(final_pred)

            y_test_i = y_test[:, i]
            y_test_i = y_test_i.reshape(len(y_test_i), 1)

            inv_new = np.concatenate((y_test_i, X_test[:, -19:]), axis=1)
            inv_new = scaler.inverse_transform(pca.inverse_transform(inv_new))
            actual_pred = inv_new[:, 0]

            y_true.append(actual_pred)
            plt.plot(final_pred[:200], label="prediction", c="b")
            plt.plot(actual_pred[:200], label="actual data", c="r")
            plt.xlim(0, 100)
            plt.ylim(0, 300)
            plt.yticks([])
            plt.xticks([])
            plt.title("comparison between prediction and actual data")
            plt.legend()
            plt.show()

            print("mean absolute error:")
            print(mean_absolute_error(final_pred, actual_pred))
            print("mean squared error:")
            print(mean_squared_error(final_pred, actual_pred))

        temp_df = pd.DataFrame(columns=['KEY', 'KM', 'DATETIME_UTC', 'PREDICTION_STEP', 'SPEED_AVG'])

        i = 0
        test_df_truncated = test_df.tail(-ahead)
        for index, row in test_df_truncated.iterrows():

            # index is datetime
            k = test_df_truncated.index.get_loc(index)
            for j in range(0, ahead):
                # print(key.split('_')[0], key.split('_')[1], index, str(j + 1), y_predicted[j][k])
                temp_df.loc[i + j] = [key.split('_')[0]] + [key.split('_')[1]] + [str(index)] + [str(j + 1)] + [
                    str(y_predicted[j][k])]
            i += 4
        result_df = result_df.append(temp_df)
        index_of_keys += 1
        if (index_of_keys % 10 == 0): result_df.to_csv('results_321_till_' + str(321 + index_of_keys) + ".csv",
                                                       encoding='utf-8', index=False)

print(trained_keys)

result_df.to_csv('results_225_till_end.csv', encoding='utf-8', index=False)