from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sb
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


FIELD_YEAR_OF_DIAGNOSIS = "A"
FIELD_YEAR_DEATH = "B"
FIELD_AGE = "C"
FIELD_SEER_CAUSE_SPECIFIC = "D"
FIELD_SEER_OTHER_CAUSE = "E"
FIELD_SURVIVAL = "F"
FIELD_COD_TO_SITE = "G"
FIELD_MONTHS_FROM_DIAG_TO_TREAT = "H"


def create_compile_net(train):
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal', input_dim=train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return NN_model

def one_hot_encoding(dataset, col):
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(dataset[[FIELD_COD_TO_SITE]])
    dataset[ohe.categories_[0]] = transformed.toarray()
    dataset[ohe.categories_[0]] = dataset[ohe.categories_[0]].astype(int)  # convert to int
    dataset.drop(FIELD_COD_TO_SITE, axis=1, inplace=True)
    return dataset


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [SURVIVAL]')
  plt.legend()
  plt.grid(True)
  plt.savefig("myeloma_plot_loss2")
  plt.close()


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
print(tf.__version__)

column_names = [FIELD_YEAR_OF_DIAGNOSIS,
                FIELD_YEAR_DEATH,
                FIELD_AGE,
                FIELD_SEER_CAUSE_SPECIFIC,
                FIELD_SEER_OTHER_CAUSE,
                FIELD_SURVIVAL,
                FIELD_COD_TO_SITE,
                FIELD_MONTHS_FROM_DIAG_TO_TREAT]

raw_dataset = pd.read_csv("myeloma_work.csv", names=column_names,
                          na_values='?', comment='\t', header=1,
                          sep=';', skipinitialspace=True)
dataset = raw_dataset.copy()
print("Dataset shape: " + str(dataset.shape))
print("Dataset head:")
print(dataset.head())
print("Null values:")
print(dataset.isna().sum())

# pandas drop empty records and colum in dataframe
dataset = dataset.dropna()

ds2 = dataset.copy()
ds2.drop(FIELD_COD_TO_SITE, axis=1, inplace=True)
ds2.hist(figsize=(12, 10))
plt.savefig("myeloma_hist_1.png")
plt.close()

C_mat = ds2.corr()
fig = plt.figure(figsize=(15,15))
sb.heatmap(C_mat, vmax=.8, square=True)
plt.savefig("myeloma_cmat_1.png")
plt.close()



# ONE HOT ENCODER COD_TO_SITE
dataset = one_hot_encoding(dataset, FIELD_COD_TO_SITE)

print("Dataset shape: " + str(dataset.shape))
print("Dataset head:")
print(dataset.head())
print("Null values:")
print(dataset.isna().sum())

dataset.hist(figsize=(12, 10))
plt.savefig("myeloma_hist_1.png")
plt.close()

dataset.describe()

# split dataset train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
train_target = train_dataset.F
train_dataset.drop(['F'], axis=1, inplace=True)

test_dataset = dataset.drop(train_dataset.index)

C_mat = train_dataset.corr()
fig = plt.figure(figsize=(15,15))
sb.heatmap(C_mat, vmax=.8, square=True)
plt.savefig("myeloma_cmat_2.png")
plt.close()

# create network and compile
model = create_compile_net(train_dataset)
model.summary()

# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint_name = 'weights_myeloma2.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
early_stopping = EarlyStopping()
callbacks_list = [checkpoint,
                  early_stopping]
history = model.fit(train_dataset, train_target,
          epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=callbacks_list)
plot_loss(history)