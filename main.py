from sklearn.datasets import make_regression, make_classification, make_blobs
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import csv          # importing the csv module

PATH_ORIG = 'myeloma_input.csv'
PATH_WORK = 'myeloma_work.csv'
FIELD_SEPARATOR = ';'

TEST_RATIO = 0.2

FIELD_YEAR_OF_DIAGNOSIS = "Year of diagnosis (1975-2019 by 5)"
FIELD_PATIENT_ID = "Patient ID"
FIELD_YEAR_FOLLOW_UP = "Year of follow-up recode"
FIELD_YEAR_DEATH = "Year of death recode"
FIELD_AGE = "Age recode with single ages and 90+"
FIELD_SEQUENCE = "Sequence number"
FIELD_SEER_CAUSE_SPECIFIC = "SEER cause-specific death classification"
FIELD_SEER_OTHER_CAUSE = "SEER other cause of death classification"
FIELD_SURVIVAL = "Survival months"
FIELD_COD_TO_SITE = "COD to site recode"
FIELD_SURVIVAL_MONTHS_FLAG =  "Survival months flag"
# COD to site rec KM
# COD to site recode ICD-0-3-2023 Revision
# COD to site recode ICD-O-3 2023 Revision Expanded (1999+)
FIELD_VITAL_STATUS = "Vital status recode (study cutoff used)"
FIELD_RADIATION = "Radiation recode"
FIELD_CHEMOTHERAPY = "Chemotherapy recode (yes, no/unk)"
FIELD_MONTHS_FROM_DIAG_TO_TREAT = "Months from diagnosis to treatment"
FIELD_PRIMARY_SITE = "Primary Site - labeled"
FIELD_HIST_BEHAV = "ICD-O-3 Hist/behav, malignant"

FIELD_N_YEAR_OF_DIAGNOSIS = 0
FIELD_N_PATIENT_ID = 1
FIELD_N_YEAR_FOLLOW_UP = 2
FIELD_N_YEAR_DEATH = 3
FIELD_N_AGE = 4
FIELD_N_SEQUENCE = 5
FIELD_N_SEER_CAUSE_SPECIFIC = 6
FIELD_N_SEER_OTHER_CAUSE = 7
FIELD_N_SURVIVAL = 8
FIELD_N_COD_TO_SITE = 9
FIELD_N_SURVIVAL_MONTHS_FLAG = 10
# 11
# 12
# 13
FIELD_N_VITAL_STATUS = 14
FIELD_N_RADIATION = 15
FIELD_N_CHEMOTHERAPY = 16
FIELD_N_MONTHS_FROM_DIAG_TO_TREAT = 17
FIELD_N_PRIMARY_SITE = 18
FIELD_N_HIST_BEHAV = 19

def add_virgolette(s):
    return '\"' + s + '\"'

def write_header(f):
    print(add_virgolette(FIELD_YEAR_OF_DIAGNOSIS), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_PATIENT_ID), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_YEAR_FOLLOW_UP), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_YEAR_DEATH), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_AGE), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_SEER_CAUSE_SPECIFIC), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_SEER_OTHER_CAUSE), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_SURVIVAL), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_COD_TO_SITE), file=f)
    # print(add_virgolette(FIELD_VITAL_STATUS), file=f, end=FIELD_SEPARATOR)
    # print(add_virgolette(FIELD_RADIATION), file=f, end=FIELD_SEPARATOR)
    # print(add_virgolette(FIELD_CHEMOTHERAPY), file=f, end=FIELD_SEPARATOR)
    # print(add_virgolette(FIELD_MONTHS_FROM_DIAG_TO_TREAT), file=f, end=FIELD_SEPARATOR)
    # print(add_virgolette(FIELD_PRIMARY_SITE), file=f, end=FIELD_SEPARATOR)
    # print(add_virgolette(FIELD_HIST_BEHAV), file=f)
    return(0)


def clean_csv(path_input, path_output):
    with open(path_input, 'r') as csv_file:
        with open(path_output, 'w') as foutput:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            header = next(csv_reader)  # store the headers and advance reader pointer
            write_header(foutput)
            for row in csv_reader:
                months_diag_treat = row[FIELD_N_MONTHS_FROM_DIAG_TO_TREAT]
                if months_diag_treat == "Blank(s)":
                    continue
                alive_or_dead = row[FIELD_N_VITAL_STATUS]
                if alive_or_dead == "Alive":
                    continue
                sequence = row[FIELD_N_SEQUENCE]
                if sequence != "1st of 2 or more primaries":
                    continue
                print(row[FIELD_N_YEAR_OF_DIAGNOSIS], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_PATIENT_ID], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_YEAR_FOLLOW_UP], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_YEAR_DEATH], end=FIELD_SEPARATOR, file=foutput)
                age = row[FIELD_N_AGE].split()[0]
                if age == "90+":
                    age = 90
                else:
                    age = int(age)
                print(age,  end=FIELD_SEPARATOR,file=foutput)
                alive_or_dead = row[FIELD_N_SEER_CAUSE_SPECIFIC].split()[0]
                if alive_or_dead == 'Alive':
                    print(0, end=FIELD_SEPARATOR, file=foutput)
                    print(1, end=FIELD_SEPARATOR, file=foutput)
                else:
                    print(1, end=FIELD_SEPARATOR, file=foutput)
                    print(0, end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_SURVIVAL], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_COD_TO_SITE], file=foutput)
                # print(row[FIELD_N_VITAL_STATUS], end=FIELD_SEPARATOR, file=foutput)
                # print(row[FIELD_N_RADIATION], end=FIELD_SEPARATOR, file=foutput)
                # print(row[FIELD_N_CHEMOTHERAPY], end=FIELD_SEPARATOR, file=foutput)
                # print(row[FIELD_N_MONTHS_FROM_DIAG_TO_TREAT], end=FIELD_SEPARATOR, file=foutput)
                # print(row[FIELD_N_PRIMARY_SITE], end=FIELD_SEPARATOR, file=foutput)
                # print(row[FIELD_N_HIST_BEHAV], file=foutput)
                line_count += 1
            print(f'Processed {line_count} lines.')


def prepare_dataset(path_input):
    with open(path_input, 'r') as csv_file:
        myeloma = pd.read_csv(path_input, sep=FIELD_SEPARATOR)
        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(myeloma[[FIELD_COD_TO_SITE]])
        myeloma[ohe.categories_[0]] = transformed.toarray()
        myeloma.drop(FIELD_COD_TO_SITE, axis=1, inplace=True)
        return myeloma

def explore_dataset(df):
    df.hist(figsize=(20,15))
    plt.savefig('myeloma_hist.png')
    print(df.head())
    df.info()
    df.describe()

    # compute correlation
    corr_matrix = df.corr()
    print(corr_matrix[FIELD_SURVIVAL].sort_values(ascending=False))

    # scatter plots
    attributes = [FIELD_SURVIVAL, FIELD_AGE, FIELD_YEAR_FOLLOW_UP, FIELD_YEAR_DEATH]
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.savefig('myeloma_scatter.png')



def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]



def main():
    np.random.seed(74)
    clean_csv(PATH_ORIG, PATH_WORK)
    myeloma = prepare_dataset(PATH_WORK)
    explore_dataset(myeloma)

    train_set, test_set = split_train_test(myeloma, TEST_RATIO)

    X = train_set
    Y = X[FIELD_SURVIVAL].values
    X = X.drop(FIELD_SURVIVAL, axis=1).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

    reg = LinearRegression()
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(reg, X_train, Y_train, cv=kfold, scoring='r2')

    print("cv_results: " + str(cv_results))
    print("mean: " + str(round(np.mean(cv_results) * 100, 2)))

if __name__ == '__main__':
    main()
