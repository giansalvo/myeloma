# VERSIONE 3
#
# MIT License
#
#   Copyright (c) 2023 Giansalvo Gusinu
#   Copyright (c) 2023 Giuseppe A. Trunfio
#   Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# adapted from https://www.tensorflow.org/tutorials/keras/regression?hl=it#regression_using_a_dnn_and_multiple_inputs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys
from hyper_param import basic_net, get_mlp_model, hyper_net

# FIELD_YEAR_OF_DIAGNOSIS = "Year of diagnosis (1975-2019 by 5)"
# FIELD_PATIENT_ID = "Patient ID"
# FIELD_YEAR_FOLLOW_UP = "Year of follow-up recode"
# FIELD_YEAR_DEATH = "Year of death recode"
# FIELD_AGE = "Age recode with single ages and 90+"
# FIELD_SEQUENCE = "Sequence number"
# FIELD_SEER_CAUSE_SPECIFIC = "SEER cause-specific death classification"
# FIELD_SEER_OTHER_CAUSE = "SEER other cause of death classification"
# FIELD_SURVIVAL = "Survival months"
# FIELD_COD_TO_SITE = "COD to site recode"
# FIELD_SURVIVAL_MONTHS_FLAG =  "Survival months flag"
# # COD to site rec KM
# # COD to site recode ICD-0-3-2023 Revision
# # COD to site recode ICD-O-3 2023 Revision Expanded (1999+)
# FIELD_VITAL_STATUS = "Vital status recode (study cutoff used)"
# FIELD_RADIATION = "Radiation recode"
# FIELD_CHEMOTHERAPY = "Chemotherapy recode (yes, no/unk)"
# FIELD_MONTHS_FROM_DIAG_TO_TREAT = "Months from diagnosis to treatment"
# FIELD_PRIMARY_SITE = "Primary Site - labeled"
# FIELD_HIST_BEHAV = "ICD-O-3 Hist/behav, malignant"

PATH_OUTPUT = "dnn"
FPATH_BENCH = os.path.join(PATH_OUTPUT, "bench.txt")
CSV_FNAME = "myeloma_work.csv"
KFOLD_NUM = 10
TEMP_WEIGHT_FNAME = 'dnn_model.keras'
RANDOM_SEED = 74
TRAIN_SIZE = 0.8
VAL_SIZE   = 0.2
TEST_SIZE  = 0.2
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 20

FIELD_YEAR_OF_DIAGNOSIS = "A"
FIELD_YEAR_DEATH = "B"
FIELD_AGE = "C"
FIELD_SEER_CAUSE_SPECIFIC = "D"
FIELD_SEER_OTHER_CAUSE = "E"
FIELD_SURVIVAL = "F"
FIELD_COD_TO_SITE = "G"
FIELD_MONTHS_FROM_DIAG_TO_TREAT = "H"

def one_hot_encoding(dataset, column):
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(dataset[[column]])
    dataset[ohe.categories_[0]] = transformed.toarray()
    dataset[ohe.categories_[0]] = dataset[ohe.categories_[0]].astype(int)  # convert to int
    dataset.drop(column, axis=1, inplace=True)
    return dataset

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

def normalization2(df):
    for col in df.columns:
        df[col] = min_max_scaling(df[col])
    return df

def denormalization(X, X_max, X_min):
    X = X * (X_max - X_min) + X_min
    return X


def build_and_compile_model(num_input):
    model = keras.Sequential([
        layers.Dense(256, input_dim=num_input, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Valid loss function for correlation: mean_squared_error, mean_absolute_error
    model.compile(loss='mean_squared_error',
                 optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=["mean_absolute_error"])
                  #, "r_squared"
    return model


def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def plot_loss(history, fname):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [SURVIVAL]')
  plt.legend()
  plt.grid(True)
  plt.savefig(fname)
  plt.close()


def plot_scatter(x, y, train_features, train_labels):
  plt.scatter(train_features['Survival'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('XXX')
  plt.ylabel('YYY')
  plt.legend()
  fname = os.path.join(PATH_OUTPUT, "myeloma_scatter_train.png")
  plt.savefig(fname)
  plt.close()

def fit_net(model, x, y, batch_size=BATCH_SIZE, val_size=VAL_SIZE, epochs=EPOCHS, patience=PATIENCE):
    cb_list = [EarlyStopping(patience=patience)]
    history = model.fit(
        x,
        y,
        batch_size=batch_size,
        callbacks=cb_list,
        validation_split=val_size,
        verbose=0, epochs=epochs,
        )
    return history


def main():
    # Make NumPy printouts easier to read.
    np.set_printoptions(precision=3, suppress=True)
    print("Python ver.: " + sys.version)
    executable = os.path.realpath(sys.executable)
    print("Python executable: " + str(executable))
    print("tensorflow version: " + tf.__version__)
    print("GPU: " + str(tf.config.list_physical_devices('GPU')))
    tf.random.set_seed(RANDOM_SEED)
    tf.config.run_functions_eagerly(False)
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    column_names = [FIELD_YEAR_OF_DIAGNOSIS,
                    FIELD_YEAR_DEATH,
                    FIELD_AGE,
                    FIELD_SEER_CAUSE_SPECIFIC,
                    FIELD_SEER_OTHER_CAUSE,
                    FIELD_SURVIVAL,
                    FIELD_COD_TO_SITE,
                    FIELD_MONTHS_FROM_DIAG_TO_TREAT]

    raw_dataset = pd.read_csv(CSV_FNAME, names=column_names,
                              na_values='?', comment='\t', header=1,
                              sep=';', skipinitialspace=True)
    dataset = raw_dataset.copy()
    print("Dataset shape: " + str(dataset.shape))
    # pandas drop empty records and colum in dataframe
    dataset = dataset.dropna()


      # ONE HOT ENCODER COD_TO_SITE
    dataset = one_hot_encoding(dataset, FIELD_COD_TO_SITE)

    dataset_norm = normalization2(dataset)
    # print(dataset_norm.describe().transpose())

    print("[NORMALIZATION]")
    # print(dataset.describe().transpose())
    # Normalization of all dataset (features and labels)
    y_min = dataset[FIELD_SURVIVAL].min()
    y_max = dataset[FIELD_SURVIVAL].max()
    dataset_norm = normalization2(dataset)
    train_features_norm = dataset_norm.sample(frac=TRAIN_SIZE, random_state=0)
    train_labels_norm = train_features_norm.pop(FIELD_SURVIVAL)

    # with open(FPATH_BENCH, "a") as f:
    #     print(str(dataset), file=f)

    model = keras.Sequential([
        layers.Dense(16, input_dim=train_features_norm.shape[1], activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Valid loss function for correlation: mean_squared_error, mean_absolute_error
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=["mean_absolute_error"])

    cb_list = [EarlyStopping(patience=PATIENCE)]
    history = model.fit(
        train_features_norm,
        train_labels_norm,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        validation_split=VAL_SIZE,
        verbose=0, epochs=EPOCHS,
    )

    fname = os.path.join(PATH_OUTPUT, "test_loss_X.png")
    plot_loss(history, fname)
    y_test = denormalization(train_labels_norm, y_max, y_min)
    res = model.evaluate(train_features_norm, y_test, batch_size=128, verbose=0)
    with open(FPATH_BENCH, "a") as f:
        print(str(res), file=f)


if __name__ == '__main__':
    main()