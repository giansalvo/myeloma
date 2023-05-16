# MIT License
#
#   Copyright (c) 2023 Giansalvo Gusinu
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
from sklearn.model_selection import ShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

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

KFOLD_NUM = 10
TEMP_WEIGHT_FNAME = 'dnn_model.keras'
RANDOM_SEED = 74
TRAIN_SIZE = 0.8
VAL_SIZE   = 0.2
TEST_SIZE  = 0.2

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


def normalization2(pd_data):
    scaler = MinMaxScaler()
    scaler.fit(pd_data)
    scaled = scaler.fit_transform(pd_data)
    scaled = pd.DataFrame(scaled, columns=pd_data.columns)
    print(scaled.head())
    return scaled


def build_and_compile_model(num_input):
  model = keras.Sequential([
      layers.Dense(64, kernel_initializer='normal', input_dim=num_input, activation='relu'),
      layers.Dense(64, kernel_initializer='normal', input_dim=num_input, activation='relu'),
      layers.Dense(64, kernel_initializer='normal', input_dim=num_input, activation='relu'),
      layers.Dense(1, activation='linear')
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

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
  plt.savefig("myeloma_scatter_train.png")
  plt.close()


def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())


def main():
    # Make NumPy printouts easier to read.
    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)
    tf.random.set_seed(RANDOM_SEED)
    tf.config.run_functions_eagerly(False)

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

    dataset.hist(figsize=(12, 10))
    plt.savefig("myeloma_hist.png")
    plt.close()

    # ONE HOT ENCODER COD_TO_SITE
    dataset = one_hot_encoding(dataset, FIELD_COD_TO_SITE)

    # Normalization
    print(dataset.describe().transpose()[['mean', 'std']])

    # split dataset train and test
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Visual inspection of the data
    sns.pairplot(train_dataset[[FIELD_YEAR_OF_DIAGNOSIS,
                                FIELD_YEAR_DEATH,
                                FIELD_AGE,
                                FIELD_SEER_CAUSE_SPECIFIC,
                                FIELD_SURVIVAL]], diag_kind='kde')
    plt.savefig("myeloma_compare.png")
    plt.close()

    print("Analytic analysis of the dataset:")
    print(train_dataset.describe().transpose())

    #  Split features from labels
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    # remove column "survival"
    train_labels = train_features.pop(FIELD_SURVIVAL)
    test_labels = test_features.pop(FIELD_SURVIVAL)
    print("train_features.head()\n" + str(train_features.head().transpose()))
    print("train_labels.head()\n" + str(train_labels.head().transpose()))

    # Normalization of input
    train_features = normalization2(train_features)
    test_features = normalization2(test_features)

    dnn_model = build_and_compile_model(train_features.shape[1])
    dnn_model.summary()

    early_stopping = EarlyStopping()

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=VAL_SIZE,
        verbose=0, epochs=200,
        callbacks=[early_stopping])
    plot_loss(history, "myeloma_plot_loss.png")

    results = []
    dnn_model.save(TEMP_WEIGHT_FNAME)
    i = 0
    ss = ShuffleSplit(n_splits=KFOLD_NUM, random_state=RANDOM_SEED, test_size=VAL_SIZE, train_size=TRAIN_SIZE)
    for i, (X_i, Y_i) in enumerate(ss.split(train_dataset)):
        print("Fold n.{} ".format(i), end="")
        # for j in range(10):
        #     print(X_i[j], end=", ")
        # print("")
        reloaded = tf.keras.models.load_model(TEMP_WEIGHT_FNAME)
        x_train = train_features.iloc[X_i]
        y_train = train_labels.iloc[X_i]
        res = reloaded.evaluate(x_train, y_train, batch_size=64, verbose=0)
        results.append(res)
        i += 1
    print("\nEvaluation on train set:")
    print("Scores: ", results)
    mean = np.mean(np.array(results))
    std = np.std(np.array(results), ddof=1)
    print("Mean: {:.4f} +/- Std:{:.4f})".format(mean, std))

    results = []
    dnn_model.save(TEMP_WEIGHT_FNAME)
    i = 0
    ss = ShuffleSplit(n_splits=KFOLD_NUM, random_state=RANDOM_SEED, test_size=TEST_SIZE, train_size=TRAIN_SIZE)
    for i, (X_i, Y_i) in enumerate(ss.split(test_dataset)):
        print("Fold n.{} ".format(i), end="")
        # for j in range(10):
        #     print(X_i[j], end=", ")
        # print("")
        reloaded = tf.keras.models.load_model(TEMP_WEIGHT_FNAME)
        x_test = test_features.iloc[Y_i]
        y_test = test_labels.iloc[Y_i]
        res = reloaded.evaluate(x_test, y_test, batch_size=128, verbose=0)
        results.append(res)
        i += 1
    print("\nEvaluation on test set:")
    print("Scores: ", results)
    mean = np.mean(np.array(results))
    std = np.std(np.array(results), ddof=1)
    print("Mean: {:.4f} +/- Std:{:.4f})".format(mean, std))

    # evaluate on test set
    #make predictions on the testset and plot
    test_predictions = dnn_model.predict(test_features).flatten()
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [Survival]')
    plt.ylabel('Predictions [Survival]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.savefig("myeloma_x_y.png")
    plt.close()

    #check the error distribution:
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Survival]')
    _ = plt.ylabel('Count')
    plt.savefig("myeloma_error_distrib.png")

    #save the model for future use
    dnn_model.save('dnn_model.keras')

if __name__ == '__main__':
    main()