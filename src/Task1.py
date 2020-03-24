from DataProc import read_process_data, categorize_mjob, compute_important_features, get_models
from DataProc import num_cols, cat_cols, equalize_test_cols, scale_numeric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor


class Task1:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 1================")
        target = 'final_score'
        train_path = 'data/assign3_students_train.txt'
        test_path = 'data/assign3_students_test.txt'
        train_data = read_process_data(train_path, clean_edu=True, clean_mjob=True)
        test_data = read_process_data(test_path, clean_edu=True, clean_mjob=True)
        train_data, test_data = scale_numeric(train_data.copy(), test_data.copy(), clean_final=False)
        train_data, test_data = equalize_test_cols(train_data.copy(), test_data.copy())
        X_train = train_data.drop(target, axis=1)
        X_test = test_data.drop(target, axis=1)
        y_train = train_data[target]
        y_test = test_data[target]
        important_cols = compute_important_features(X_train, y_train, classifier=False, k=35)
        # print(important_cols)
        X_train = X_train[important_cols]
        X_test = X_test[important_cols]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.m1_mse = mean_squared_error(y_test, pred)

        model = AdaBoostRegressor(n_estimators=25)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.m2_mse = mean_squared_error(y_test, pred)

        return

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.

        # Evaluate learned model on testing data, and print the results.
        print("Mean squared error\t" + str(round(self.m1_mse, 2)))
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.

        # Evaluate learned model on testing data, and print the results.
        print("Mean squared error\t" + str(round(self.m2_mse, 2)))
        return
