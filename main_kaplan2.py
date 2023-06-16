"""
    Models for estimate the survival time period for Myeloma affected patients

    Copyright (c) 2023 Giansalvo Gusinu

    Code adapted from following articles/repositories:
    https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
    https://github.com/dhassault/tf-semantic-example

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, PredictionErrorDisplay, RocCurveDisplay, mean_squared_error, auc, r2_score
from sklearn.svm import LinearSVR, SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import *
from sklearn.cross_decomposition import *
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import csv          # importing the csv module
import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import (logrank_test,
                                  pairwise_logrank_test,
                                  multivariate_logrank_test,
                                  survival_difference_at_fixed_point_in_time_test)


PATH_ORIG = 'myeloma_kaplan.csv'
PATH_WORK = 'myeloma_kaplan_clean.csv'
FIELD_SEPARATOR = ';'
PATH_PRED_ERR = "kaplan"

TEST_RATIO = 0.2
N_SPLIT = 10

FIELD_SEX = "Sex"
FIELD_YEAR_OF_DIAGNOSIS = "Year of diagnosis"
FIELD_RACE = "Race recode (W, B, AI, API)"
FIELD_DIAG_CONFIRM = "Diagnostic Confirmation"
FIELD_AJCC = "AJCC ID (2018+)"
FIELD_MALIGNANT = "First malignant primary indicator"
FIELD_TOT_TUMORS = "Total number of in situ/malignant tumors for patient"
FILD_AGE = "Age recode with single ages and 90+"
FIELD_YEAR_DEATH = "Year of death recode"


# INPUT
FIELD_N_SEX = 0
FIELD_N_YEAR_OF_DIAGNOSIS = 1
FIELD_N_RACE = 2
FIELD_N_DIAG_CONFIRM = 3
FIELD_N_AJCC = 4
FIELD_N_MALIGNANT = 6
FIELD_N_TOT_TUMORS = 7
FIELD_N_AGE = 9
FIELD_N_YEAR_DEATH = 10

def add_virgolette(s):
    return '\"' + s + '\"'

def write_header(f):
    print(add_virgolette("time"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("died"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("race"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_DIAG_CONFIRM), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_MALIGNANT), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_AJCC), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("age_class"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("sex"), file=f)
    FIELD_DIAG_CONFIRM

    return(0)


def clean_csv(path_input, path_output):
    with open(path_input, 'r') as csv_file:
        with open(path_output, 'w') as foutput:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            header = next(csv_reader)  # store the headers and advance reader pointer
            write_header(foutput)
            for row in csv_reader:
                #################################
                # time and death_flag
                #################################
                year_diag = row[FIELD_N_YEAR_OF_DIAGNOSIS]
                year_death = row[FIELD_N_YEAR_DEATH]
                if (year_diag == "Patient ID") or (year_death == "Year of death recode"):
                    # bug in the csv file
                    continue
                if year_death == "Alive at last contact":
                    time = 0
                    death_flag = 0
                else:
                    year_diag = int(year_diag)
                    year_death = int(year_death)
                    if year_diag > year_death:
                        # bug
                        # print("Error: year_diag > year_death: {} > {}".format(year_diag, year_death))
                        continue
                    if year_diag < 1975 or year_diag > 2023:
                        # print("Error: year_diag out of range {}".format(year_diag))
                        continue
                    if year_death < 1975 or year_death > 2023:
                        # print("Error: year_death out of range {}".format(year_death))
                        continue
                    time = (year_death - year_diag)
                    death_flag = 1
                #################################
                # time and death_flag
                #################################
                sex = row[FIELD_N_SEX]
                if sex != "Male" and sex != "Female":
                    # bug
                    # print("Error: sex out of range")
                    continue
                #################################
                # time and death_flag
                #################################
                age = row[FIELD_N_AGE].split()[0]
                if age == "90+":
                    age_class = "90+"
                else:
                    age = int(age)
                    if age < 10:
                        age_class = "0-9 years"
                    elif age < 20:
                        age_class = "10-19 years"
                    elif age < 30:
                        age_class = "20-29 years"
                    elif age < 40:
                        age_class = "30-39 years"
                    elif age < 50:
                        age_class = "40-49 years"
                    elif age < 60:
                        age_class = "50-59 years"
                    elif age < 70:
                        age_class = "60-69 years"
                    elif age < 80:
                        age_class = "70-79 years"
                    elif age < 90:
                        age_class = "80-89 years"

                diag_confirm = row[FIELD_N_DIAG_CONFIRM]
                malignant = row[FIELD_N_MALIGNANT]
                ajcc = row[FIELD_N_AJCC]
                race = row[FIELD_N_RACE]

                # print record
                print(str(time), end=FIELD_SEPARATOR, file=foutput)
                print(str(death_flag), end=FIELD_SEPARATOR, file=foutput)
                print(race, end=FIELD_SEPARATOR, file=foutput)
                print(diag_confirm, end=FIELD_SEPARATOR, file=foutput)
                print(malignant, end=FIELD_SEPARATOR, file=foutput)
                print(ajcc, end=FIELD_SEPARATOR, file=foutput)
                print(age_class, end=FIELD_SEPARATOR, file=foutput)
                print(sex, file=foutput)

                line_count += 1
            print(f'Processed {line_count} lines.')



def main():
    np.random.seed(74)
    if not os.path.exists(PATH_PRED_ERR):
        os.makedirs(PATH_PRED_ERR)
    clean_csv(PATH_ORIG, PATH_WORK)

    df = pd.read_csv(PATH_WORK, sep=FIELD_SEPARATOR)
    # df['churn'] = [1 if x == 'Yes' else 0 for x in df['Churn']]
    print(df.head)

    T = df['time']
    E = df['died']

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    for s in df['sex'].unique():
        flag = df['sex'] == s
        kmf.fit(T[flag], event_observed=E[flag], label=s)
        kmf.plot_survival_function(ax=ax, ci_show=False, loc=slice(0.,10.))
    plt.title("Survival curves by Sex")
    plt.show()
    plt.close()
    print(kmf.median_survival_time_)

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    for s in df['race'].unique():
        flag = df['race'] == s
        kmf.fit(T[flag], event_observed=E[flag], label=s)
        kmf.plot_survival_function(ax=ax, ci_show=False, loc=slice(0.,10.))
    plt.title("Survival curves by Race")
    plt.show()
    plt.close()
    print(kmf.median_survival_time_)

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    for s in df[FIELD_MALIGNANT].unique():
        flag = df[FIELD_MALIGNANT] == s
        kmf.fit(T[flag], event_observed=E[flag], label=s)
        kmf.plot_survival_function(ax=ax, ci_show=False, loc=slice(0.,10.))
    plt.title("Survival curves by First Malignant")
    plt.show()
    plt.close()
    print(kmf.median_survival_time_)

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    for s in df["age_class"].unique():
        flag = df["age_class"] == s
        kmf.fit(T[flag], event_observed=E[flag], label=s)
        kmf.plot_survival_function(ax=ax, ci_show=False, loc=slice(0.,10.))
    plt.title("Survival curves by Age Class")
    plt.show()
    plt.close()
    print(kmf.median_survival_time_)

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    for s in df[FIELD_AJCC].unique():
        flag = df[FIELD_AJCC] == s
        kmf.fit(T[flag], event_observed=E[flag], label=s)
        kmf.plot_survival_function(ax=ax, ci_show=False, loc=slice(0.,10.))
    plt.title("Survival curves by AJCC")
    plt.legend(bbox_to_anchor=(0, 0), loc="upper center", mode="expand", ncol=1)
    plt.tight_layout()
    plt.show()
    plt.close()
    print(kmf.median_survival_time_)

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    for s in df[FIELD_DIAG_CONFIRM].unique():
        flag = df[FIELD_DIAG_CONFIRM] == s
        kmf.fit(T[flag], event_observed=E[flag], label=s)
        kmf.plot_survival_function(ax=ax, ci_show=False, loc=slice(0.,10.))
    # place legend below plot
    plt.legend(bbox_to_anchor=(0, 0), loc="upper center", mode="expand", ncol=1)
    plt.tight_layout()
    plt.title("Survival curves by Diagnostic Confirmation")
    plt.show()
    plt.close()
    print(kmf.median_survival_time_)


if __name__ == '__main__':
    main()
