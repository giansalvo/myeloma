# adapted from
# https://pyimagesearch.com/2021/05/31/hyperparameter-tuning-for-deep-learning-with-scikit-learn-keras-and-tensorflow/

# import the necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV

def get_mlp_model(hiddenLayerOne=784, hiddenLayerTwo=256,
    dropout=0.2, learnRate=0.01):

	# initialize a sequential model and add layer to flatten the
	# input data
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(hiddenLayerOne, activation="relu", input_shape=(784,)))

    # add two stacks of FC => RELU => DROPOUT
    model.add(Dense(hiddenLayerOne, activation="relu", input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(hiddenLayerTwo, activation="relu"))
    model.add(Dropout(dropout))

    # add a sigmoid activation layer for output
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(
        optimizer=Adam(learning_rate=learnRate),
        loss="mean_squared_error",
        metrics=["accuracy"])
    # return compiled model
    return model

def basic_net(trainData, trainLabels, testData, testLabels):
    # initialize our model with the default hyperparameter values
    print("[INFO] initializing model...")
    model = get_mlp_model()
    # train the network (i.e., no hyperparameter tuning)
    print("[INFO] training model...")
    cb_list = [EarlyStopping(patience=20)]
    history = model.fit(x=trainData, y=trainLabels,
                  validation_data=(testData, testLabels),
                  verbose=0,
                  batch_size=64,
                  callbacks=cb_list,
                  epochs=200)
    # make predictions on the test set and evaluate it
    print("[INFO] evaluating network...")
    score = model.evaluate(testData, testLabels)
    print(type(score))
    print(str(score))
    # print("score: {:.4f}".format(score))[1]
    return history

def hyper_net(trainData, trainLabels, testData, testLabels):
    print("[INFO] initializing model...")
    model = KerasRegressor(build_fn=get_mlp_model, verbose=0)
    # define a grid of the hyperparameter search space
    hiddenLayerOne = [8, 32, 64]
    hiddenLayerTwo = [8, 32, 64]
    learnRate = [1e-2, 1e-3, 1e-4]
    dropout = [0.3, 0.4, 0.5]
    batchSize = [4, 8, 16]
    epochs = [10, 20, 30, 40]
    # create a dictionary from the hyperparameter grid
    grid = dict(
        hiddenLayerOne=hiddenLayerOne,
        learnRate=learnRate,
        hiddenLayerTwo=hiddenLayerTwo,
        dropout=dropout,
        batch_size=batchSize,
        epochs=epochs
    )
    # initialize a random search with a 3-fold cross-validation and then
    # start the hyperparameter search process
    print("[INFO] performing random search...")
    searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
        param_distributions=grid, scoring="neg_mean_squared_error")
    searchResults = searcher.fit(trainData, trainLabels)
    # summarize grid search information
    bestScore = searchResults.best_score_
    bestParams = searchResults.best_params_
    print("[INFO] best score on train data is {:.2f} using {}".format(bestScore,
        bestParams))

    # extract the best model, make predictions on our data, and show a
    # classification report
    print("[INFO] evaluating the best model on test set...")
    bestModel = searchResults.best_estimator_
    score = bestModel.score(testData, testLabels)
    print("score: {:.4f}".format(score))