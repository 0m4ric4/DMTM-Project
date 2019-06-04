from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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




dataframe_dict= pd.read_csv('dictionary.csv')
dictionary = {}
for index,row in dataframe_dict.iterrows():
    dictionary[row['EVENTO']] = row['TIPO']
print(dictionary)

#train = pd.read_csv('train_4_june.csv.gz',dtype=dictionary)
#test = pd.read_csv('test_4_june.csv.gz',dtype=dictionary)



train_df = pd.read_csv('train_df_4_june.csv')
test_df = pd.read_csv('test_df_4_june.csv')

print(train_df.info())
print(train_df.head(20))

print(train_df.info())
print(train_df.head(20))

values_train = train_df.values

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(values_train)

# print(scaled_train.head())


# test_df.set_index('DATETIME_UTC', inplace=True)


print(test_df.info())
print(test_df.head(20))

values_test = test_df.values

scaler = MinMaxScaler()
scaled_test = scaler.fit_transform(values_test)

# print(scaled_test.head())

timeSteps = 1
ahead = 4

supervised_train = series_to_supervised(scaled_train, n_in=timeSteps, n_out=ahead)
supervised_test = series_to_supervised(scaled_test, n_in=timeSteps, n_out=ahead)

#df = supervised_train[supervised_train.columns.drop(list(supervised_train.filter(regex='Test')))]



features_train = train_df.shape[1]
features_test = test_df.shape[1]


'''
for i in range(0,timeSteps):
    n = (features_train*2)+i
    supervised_train.drop(supervised_train.columns[range(features_train+1, n)], axis=1, inplace=True)
    supervised_test.drop(supervised_test.columns[range(features_train+1, n)], axis=1, inplace=True)

'''

supervised_train.drop(supervised_train.columns[range(124,246)], axis=1, inplace=True)
supervised_train.drop(supervised_train.columns[range(125,247)], axis=1, inplace=True)
supervised_train.drop(supervised_train.columns[range(126,248)], axis=1, inplace=True)
supervised_train.drop(supervised_train.columns[range(127,249)], axis=1, inplace=True)
#supervised_train.drop(supervised_train.columns[range(171,340)], axis=1, inplace=True)

supervised_test.drop(supervised_test.columns[range(124,246)], axis=1, inplace=True)
supervised_test.drop(supervised_test.columns[range(125,247)], axis=1, inplace=True)
supervised_test.drop(supervised_test.columns[range(126,248)], axis=1, inplace=True)
supervised_test.drop(supervised_test.columns[range(127,249)], axis=1, inplace=True)

supervised_train = supervised_train.values
supervised_test = supervised_test.values
#X_train = supervised_train[:, :-ahead]
#y_train = supervised_train[:, -ahead:-1]

X_train = supervised_train[:, :features_train * timeSteps]
y_train = supervised_train[:, features_train * timeSteps:]

X_test = supervised_test[:, :features_test * timeSteps]
y_test = supervised_test[:, features_test * timeSteps:]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = X_train.reshape(X_train.shape[0], timeSteps, features_train)
X_test = X_test.reshape(X_test.shape[0], timeSteps, features_test)



print(X_train.shape, X_test.shape)


'''
model = Sequential()
model.add( LSTM(30, input_shape = ( timeSteps,X_train.shape[2]) ) )
model.add( Dense(ahead) )

model.compile( loss = "mae", optimizer = "adam")

history =  model.fit( X_train,y_train, validation_data = (X_test,y_test), epochs = 20 , batch_size = 4, verbose = 2, shuffle = False)
'''
NUM_NEURONS_FirstLayer = 100
NUM_NEURONS_SecondLayer = 50
EPOCHS = 50

#Build the model
model = Sequential()
model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(timeSteps,X_train.shape[2]), return_sequences=True))
model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,1)))

model.add(Dense(ahead))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train,y_train,epochs=EPOCHS,validation_data=(X_test,y_test),shuffle=True,batch_size=32, verbose=2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.yticks([])
plt.xticks([])
plt.title("loss during training")
plt.show()


y_pred = model.predict(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[2] * X_test.shape[1])

for i in range(0,ahead):

    y_pred_i = y_pred[:, i]
    y_pred_i = y_pred_i.reshape(y_test.shape[0], 1)

    inv_new = np.concatenate((y_pred_i, X_test[:, -122:]), axis=1)
    inv_new = scaler.inverse_transform(inv_new)
    final_pred = inv_new[:, 0]

    y_test_i = y_test[:, i]
    y_test_i = y_test_i.reshape(len(y_test_i), 1)

    inv_new = np.concatenate((y_test_i, X_test[:, -122:]), axis=1)
    inv_new = scaler.inverse_transform(inv_new)
    actual_pred = inv_new[:, 0]

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

print("...")
# pd.DataFrame(supervised).head()
