"""
    Models for estimate the survival time period for Myeloma affected patients

    Copyright (c) 2023 Giansalvo Gusinu

    Code adapted from following articles/repositories:
    https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
    https://github.com/dhassault/tf-semantic-example
    https://stackoverflow.com/questions/46852222/how-can-i-import-all-of-sklearns-regressors

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
from sklearn.linear_model import *
from sklearn.multioutput import *
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, PredictionErrorDisplay, RocCurveDisplay, mean_squared_error, auc, r2_score
from sklearn.metrics import get_scorer_names, mean_absolute_error, median_absolute_error, roc_auc_score, r2_score, explained_variance_score
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import *
from sklearn.ensemble import *
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

PATH_INPUT = '20230515_v6.csv'
PATH_OUTPUT = 'myeloma_cvs6.csv'
PATH_BENCHMARK = 'myeloma_benchmark.csv'
FIELD_SEPARATOR = ';'

TEST_RATIO = 0.2
N_SPLIT = 5

FIELD_AGE = "Age recode with single ages and 90+"
FIELD_RACE = "Race recode (W, B, AI, API)"
FIELD_SEX = "Sex"
FIELD_EMD = "EMD"
FIELD_EMD_SITE = "EMD Site - labeled"
FIELD_COD_TO_SITE = "COD to site recode"
FIELD_SEER_SPECIFIC = "SEER cause-specific death classification"
FIELD_SURVIVAL_1 = "survival month (1st Dx)"
FIELD_SURVIVAL_2 = "survival month (2nd EMD)"

FIELD_N_AGE = 0
FIELD_N_RACE = 1
FIELD_N_SEX = 2
FIELD_N_EMD = 3
FIELD_N_EMD_SITE = 4
FIELD_N_COD_TO_SITE = 5
FIELD_N_SEER_SPECIFIC = 6
FIELD_N_SURVIVAL_1 = 7
FIELD_N_SURVIVAL_2 = 8

def regression_roc_auc_score(y_true, y_pred, num_rounds=10000):
    """
    Computes Regression-ROC-AUC-score.

    Parameters:
    ----------
    y_true: array-like of shape (n_samples,). Binary or continuous target variable.
    y_pred: array-like of shape (n_samples,). Target scores.
    num_rounds: int or string. If integer, number of random pairs of observations.
                If string, 'exact', all possible pairs of observations will be evaluated.

    Returns:
    -------
    rroc: float. Regression-ROC-AUC-score.
    """

    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_pairs = 0
    num_same_sign = 0

    for i, j in _yield_pairs(y_true, num_rounds):
        diff_true = y_true[i] - y_true[j]
        diff_score = y_pred[i] - y_pred[j]
        if diff_true * diff_score > 0:
            num_same_sign += 1
        elif diff_score == 0:
            num_same_sign += .5
        num_pairs += 1

    return num_same_sign / num_pairs


def _yield_pairs(y_true, num_rounds):
    """
    Returns pairs of valid indices. Indices must belong to observations having different values.

    Parameters:
    ----------
    y_true: array-like of shape (n_samples,). Binary or continuous target variable.
    num_rounds: int or string. If integer, number of random pairs of observations to return.
                If string, 'exact', all possible pairs of observations will be returned.

    Yields:
    -------
    i, j: tuple of int of shape (2,). Indices referred to a pair of samples.

    """
    import numpy as np

    if num_rounds == 'exact':
        for i in range(len(y_true)):
            for j in np.where((y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
                yield i, j
    else:
        for r in range(num_rounds):
            i = np.random.choice(range(len(y_true)))
            j = np.random.choice(np.where(y_true != y_true[i])[0])
            yield i, j


def add_virgolette(s):
    return '\"' + s + '\"'

def write_header(f):
    print(add_virgolette(FIELD_AGE), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_RACE), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_SEX), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_EMD), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_EMD_SITE), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_COD_TO_SITE), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_SEER_SPECIFIC), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_SURVIVAL_1), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette(FIELD_SURVIVAL_2), file=f)
    return(0)


def clean_csv(path_input, path_output):
    with open(path_input, 'r') as csv_file:
        with open(path_output, 'w') as foutput:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            header = next(csv_reader)  # store the headers and advance reader pointer
            write_header(foutput)
            for row in csv_reader:
                age = row[FIELD_N_AGE]
                age = age.split()[0]
                if age == "90+":
                    age = 90
                else:
                    age = int(age)
                print(age,  end=FIELD_SEPARATOR,file=foutput)
                print(row[FIELD_N_RACE], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_SEX], end=FIELD_SEPARATOR, file=foutput)

                print(row[FIELD_N_EMD], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_EMD_SITE], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_COD_TO_SITE], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_SEER_SPECIFIC], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_SURVIVAL_1], end=FIELD_SEPARATOR, file=foutput)
                print(row[FIELD_N_SURVIVAL_2], file=foutput)
                line_count += 1
            print(f'Processed {line_count} lines.')


def prepare_dataset(path_input):
    with open(path_input, 'r') as csv_file:
        myeloma = pd.read_csv(path_input, sep=FIELD_SEPARATOR)

        ######################################################################
        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(myeloma[[FIELD_COD_TO_SITE]])
        myeloma[ohe.categories_[0]] = transformed.toarray()
        myeloma[ohe.categories_[0]] = myeloma[ohe.categories_[0]].astype(int)   # convert to int
        myeloma.drop(FIELD_COD_TO_SITE, axis=1, inplace=True)

        counts = []
        names = []
        for cat in enumerate(ohe.categories_[0]):
            label = cat[1]
            names.append(label)
            counts.append(myeloma[label].value_counts()[1])

        fig, ax = plt.subplots()
        bar_container = ax.bar(names, counts)
        ax.set(ylabel='count', title="COD to site")
        ax.bar_label(bar_container, fmt='{:,.0f}')
        plt.savefig("myel_hist_COD.png")
        plt.close()

        ######################################################################
        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(myeloma[[FIELD_RACE]])
        myeloma[ohe.categories_[0]] = transformed.toarray()
        myeloma[ohe.categories_[0]] = myeloma[ohe.categories_[0]].astype(int)   # convert to int
        myeloma.drop(FIELD_RACE, axis=1, inplace=True)

        counts = []
        names = []
        for cat in enumerate(ohe.categories_[0]):
            label = cat[1]
            names.append(label)
            counts.append(myeloma[label].value_counts()[1])

        fig, ax = plt.subplots()
        # counts = [r1, r2, r3, r4, r5]
        # names = ["C40.2", "C41.0", "41.2", "C41.4", "C41.9"]
        bar_container = ax.bar(names, counts)
        ax.set(ylabel='count', title="Race")
        ax.bar_label(bar_container, fmt='{:,.0f}')
        plt.savefig("myel_hist_race.png")
        plt.close()


        ######################################################################
        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(myeloma[[FIELD_SEX]])
        myeloma[ohe.categories_[0]] = transformed.toarray()
        myeloma[ohe.categories_[0]] = myeloma[ohe.categories_[0]].astype(int)  # convert to int
        myeloma.drop(FIELD_SEX, axis=1, inplace=True)

        counts = []
        names = []
        for cat in enumerate(ohe.categories_[0]):
            label = cat[1]
            names.append(label)
            counts.append(myeloma[label].value_counts()[1])

        fig, ax = plt.subplots()
        bar_container = ax.bar(names, counts)
        ax.set(ylabel='count', title="SEX")
        ax.bar_label(bar_container, fmt='{:,.0f}')
        plt.savefig("myel_hist_sex.png")
        plt.close()

        ######################################################################
        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(myeloma[[FIELD_EMD]])
        myeloma[ohe.categories_[0]] = transformed.toarray()
        myeloma[ohe.categories_[0]] = myeloma[ohe.categories_[0]].astype(int)  # convert to int
        myeloma.drop(FIELD_EMD, axis=1, inplace=True)

        counts = []
        names = []
        for cat in enumerate(ohe.categories_[0]):
            label = cat[1]
            names.append(label)
            counts.append(myeloma[label].value_counts()[1])

        fig, ax = plt.subplots()
        bar_container = ax.bar(names, counts)
        ax.set(ylabel='count', title="EMD")
        ax.bar_label(bar_container, fmt='{:,.0f}')
        plt.savefig("myel_hist_EMD.png")
        plt.close()

        ######################################################################
        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(myeloma[[FIELD_EMD_SITE]])
        myeloma[ohe.categories_[0]] = transformed.toarray()
        myeloma[ohe.categories_[0]] = myeloma[ohe.categories_[0]].astype(int)  # convert to int
        myeloma.drop(FIELD_EMD_SITE, axis=1, inplace=True)

        counts = []
        names = []
        for cat in enumerate(ohe.categories_[0]):
            label = cat[1]
            names.append(label)
            counts.append(myeloma[label].value_counts()[1])

        fig, ax = plt.subplots()
        # counts = [r1, r2, r3, r4, r5]
        # names = ["C40.2", "C41.0", "41.2", "C41.4", "C41.9"]
        bar_container = ax.bar(names, counts)
        ax.set(ylabel='count', title="EMD Site")
        ax.bar_label(bar_container, fmt='{:,.0f}')
        plt.savefig("myel_hist_EMD_site.png")
        plt.close()


        ######################################################################
        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(myeloma[[FIELD_SEER_SPECIFIC]])
        myeloma[ohe.categories_[0]] = transformed.toarray()
        myeloma[ohe.categories_[0]] = myeloma[ohe.categories_[0]].astype(int)  # convert to int
        myeloma.drop(FIELD_SEER_SPECIFIC, axis=1, inplace=True)

        counts = []
        names = []
        for cat in enumerate(ohe.categories_[0]):
            label = cat[1]
            names.append(label)
            counts.append(myeloma[label].value_counts()[1])

        fig, ax = plt.subplots()
        bar_container = ax.bar(names, counts)
        ax.set(ylabel='count', title="SEER Specific")
        ax.bar_label(bar_container, fmt='{:,.0f}')
        plt.savefig("myel_hist_SEER_specific.png")
        plt.close()

        return myeloma


def plot_graph(df, title, fname):
    plt.title(title, fontsize=40)
    df[title].hist(figsize=(20, 15))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(fname)
    plt.close()


def explore_dataset(df):
    plot_graph(df, FIELD_AGE, 'myel_hist_age.png')
    plot_graph(df, FIELD_SURVIVAL_1, 'myel_hist_survival1.png')
    plot_graph(df, FIELD_SURVIVAL_2, 'myel_hist_survival2.png')
    # plot_sex(df, title="Sex", fname="myel_hist_sex.png")
    # # plot_race(df, title="Race", fname="myel_hist_race.png")
    # plot_EMD(df, title="EMD", fname="myel_hist_emd.png")

    df.info()
    df.describe()

    # compute correlation
    corr_matrix = df.corr()
    print(corr_matrix[FIELD_SURVIVAL_1].sort_values(ascending=False))
    # scatter plots
    attributes = [FIELD_AGE, FIELD_SURVIVAL_1, FIELD_SURVIVAL_2]
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.savefig('myeloma_scatter1.png')
    print(df.value_counts())


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def display_scores(scores):
        print("Scores: ", np.round(scores, 4))
        print("Mean: ", np.round(scores.mean(), 4))
        print("Standard deviation: ", np.round(scores.std(), 4))

def display_pred_error(regressor, X_train, Y_train, kfold, fname):
        y_pred = cross_val_predict(regressor, X_train, Y_train, cv=kfold)
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            Y_train,
            y_pred=y_pred,
            kind="actual_vs_predicted",
            subsample=100,
            ax=axs[0],
            random_state=0,
        )
        axs[0].set_title("Actual vs. Predicted values")
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        PredictionErrorDisplay.from_predictions(
            Y_train,
            y_pred=y_pred,
            kind="residual_vs_predicted",
            subsample=100,
            ax=axs[1],
            random_state=0,
        )
        axs[1].set_title("Residuals vs. Predicted Values")
        fig.suptitle("Plotting cross-validated predictions")
        plt.tight_layout()
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.savefig(fname)
        plt.close()


def evaluate_regressor(reg, X_train, Y_train, scoring, fname):
    # scoring = {'accuracy': 'accuracy',
    #            'recall': 'recall',
    #            'precision': 'precision',
    #            'roc_auc': 'roc_auc'}
    # cross_validate...
    kfold = KFold(n_splits=N_SPLIT)
    scores = cross_val_score(reg, X_train, Y_train, cv=kfold, scoring=scoring)
    display_scores(scores)
    display_pred_error(reg, X_train, Y_train, kfold, fname)


def main():
    np.random.seed(74)

    clean_csv(PATH_INPUT, PATH_OUTPUT)
    myeloma = prepare_dataset(PATH_OUTPUT)
    explore_dataset(myeloma)

    train_set, test_set = split_train_test(myeloma, TEST_RATIO)

    X = train_set
    Y = X[FIELD_SURVIVAL_2].values
    X = X.drop(FIELD_SURVIVAL_2, axis=1).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
    print("X_train, X_test, Y_train, Y_test",
          X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    #########################################
    # linear regression
    print("\nLinearRegression")
    reg = LinearRegression()
    evaluate_regressor(reg, X_train, Y_train, 'r2','myeloma_pred_err_lr.png')

    # evaluate on test set this regressor
    print("\nEvaluate on testset")
    reg.fit(X_train, Y_train)
    pred_testset = reg.predict(X_test)
    testset_mse = mean_squared_error(Y_test, pred_testset)
    testset_rmse = np.sqrt(testset_mse)
    print(f"testset_rmse: {testset_rmse:.2f} months")

    # Plot outputs
    plt.close()
    plt.scatter(Y_test, pred_testset, color="blue")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig("myeloma_pred_scattered_LR.png")

    #########################################
    # DecisionTreeRegressor
    print("\nDecisionTreeRegression")
    reg = DecisionTreeRegressor()
    evaluate_regressor(reg, X_train, Y_train, 'r2', 'myeloma_pred_err_DTR.png')
    # save the model on file

    # evaluate on test set this regressor
    print("\nEvaluate on testset")
    reg.fit(X_train, Y_train)
    pred_testset = reg.predict(X_test)
    testset_mse = mean_squared_error(Y_test, pred_testset)
    testset_rmse = np.sqrt(testset_mse)
    print(f"testset_rmse: {testset_rmse:.2f} months")

    # Plot outputs
    plt.close()
    plt.scatter(Y_test, pred_testset, color="blue")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig("myeloma_pred_scattered_DT.png")

    #########################################
    # RandomForestRegressor
    print("\nRandomForestRegressor")
    reg = RandomForestRegressor()
    evaluate_regressor(reg, X_train, Y_train, 'r2', 'myeloma_pred_err_RF.png')

    # evaluate on test set last regressor
    print("\nEvaluate on testset")
    reg.fit(X_train, Y_train)
    pred_testset = reg.predict(X_test)
    testset_mse = mean_squared_error(Y_test, pred_testset)
    testset_rmse = np.sqrt(testset_mse)
    print(f"testset_rmse: {testset_rmse:.2f} months")

    # Plot outputs
    plt.close()
    plt.scatter(Y_test, pred_testset, color="blue")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig("myeloma_pred_scattered_RF.png")


    # inspired by https://medium.com/towards-data-science/how-to-calculate-roc-auc-score-for-regression-models-c0be4fdf76bb
    modelnames = [
        "ARDRegression()",
        "AdaBoostRegressor()",
        "BaggingRegressor()",
        "BayesianRidge()",
        # "CCA()",
        "DecisionTreeRegressor()",
        "ElasticNet()",
        "ElasticNetCV()",
        "ExtraTreeRegressor()",
        # "GammaRegressor()",
        "GaussianProcessRegressor()",
        "GradientBoostingRegressor()",
        "HistGradientBoostingRegressor()",
        # "HuberRegressor()",
        # "IsotonicRegression()",
        "KNeighborsRegressor()",
        "KernelRidge()",
        # "Lars()",
        # "LarsCV()",
        "Lasso()",
        "LassoCV()",
        # "LassoLars()",
        # "LassoLarsCV()",
        # "LassoLarsIC()",
        "LinearRegression()",
        # "LinearSVR()",
        # "MLPRegressor(hidden_layer_sizes=(16, 16))",
        # "MultiOutputRegressor()",
        # "MultiTaskElasticNet()",
        # "MultiTaskElasticNetCV()",
        # "MultiTaskLasso()",
        # "MultiTaskLassoCV()",
        "NuSVR()",
        "OrthogonalMatchingPursuit()",
        # "OrthogonalMatchingPursuitCV()",
        # "PLSCanonical()",
        # "PLSRegression()",
        "PassiveAggressiveRegressor()",
        # "PoissonRegressor()",
        # "QuantileRegressor()",
        "RANSACRegressor()",
        "RadiusNeighborsRegressor()",
        "RandomForestRegressor()",
        # "RegressorChain()",
        "Ridge()",
        # "RidgeCV()",
        "SGDRegressor()",
        "SVR()",
        # "StackingRegressor()",
        "TheilSenRegressor()",
        "TransformedTargetRegressor()",
        # "TweedieRegressor()",
        # "VotingRegressor()"
    ]

    metricnames = [
        "mean_absolute_error",      # MAE
        "median_absolute_error",    # MedAE
        "r2_score",                 # r2
        "explained_variance_score",
        "regression_roc_auc_score"
    ]
    metrics = pd.DataFrame(index=modelnames, columns=metricnames)
    # Compute benchmarks
    for modelname in modelnames:
        print("Fitting model: ", modelname)
        model = eval(modelname)
        pred_test = model.fit(X_train, Y_train).predict(X_test)
        for metricname in metricnames:
            metrics.loc[modelname, metricname] = eval(f'{metricname}(Y_test, pred_test)')
            kfold = KFold(n_splits=N_SPLIT)
            # scores = cross_val_score(reg, X_train, Y_train, cv=kfold, scoring=metricname)
            # display_scores(scores)
            fname = "pred_err_" + modelname + "_" + metricname + ".png"
            display_pred_error(model, X_train, Y_train, kfold, fname)

    # print results of benchmarks
    with open(PATH_BENCHMARK, "w") as f:
        # print header
        print("Regressor Model; ", file=f, end="")
        for metricname in metricnames:
            print(metricname + FIELD_SEPARATOR, file=f, end="")
        print("", file=f, end="\n")
        for modelname in modelnames:
            print(modelname + FIELD_SEPARATOR, file=f, end=" ")
            for metricname in metricnames:
                # fname = str(modelname)+"_" + str(metricname)
                # evaluate_regressor(modelname, X_train, Y_train, metricname, fname)
                val = metrics.loc[modelname][metricname]
                print(format(val).replace('.', ',') + FIELD_SEPARATOR, file=f, end=" ")
            print("", file=f)


if __name__ == '__main__':
    main()
