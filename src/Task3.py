import warnings
warnings.filterwarnings('ignore')
from DataProc import read_process_data, categorize_mjob, compute_important_features, get_models
from DataProc import  num_cols, cat_cols, equalize_test_cols, scale_numeric, compute_edu_important
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import hamming_loss


class Task3:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 3================")
        target = ['schoolsup', 'familysup', 'paidsup', 'nosup']
        train_path = 'data/assign3_students_train.txt'
        test_path = 'data/assign3_students_test.txt'
        train_data = read_process_data(train_path, clean_edu=True, clean_mjob=True)
        test_data = read_process_data(test_path, clean_edu=True, clean_mjob=True)
        train_data, test_data = scale_numeric(train_data.copy(), test_data.copy(), clean_final=True)
        train_data, test_data = equalize_test_cols(train_data.copy(), test_data.copy())
        X_train = train_data.drop(target, axis=1)
        X_test = test_data.drop(target, axis=1)
        y_train = train_data[target]
        y_test = test_data[target]
        imp_cols = compute_edu_important(X_train, y_train, k=80, cols=['schoolsup', 'familysup', 'paidsup', 'nosup'])
        X_train = X_train[imp_cols]
        X_test = X_test[imp_cols]

        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        self.model_1_accuracy = clf.score(X_test, y_test)
        pred = clf.predict(X_test)
        self.model_1_hamming_loss = hamming_loss(y_test, pred)

        clf = MultiOutputClassifier(GradientBoostingClassifier())
        clf.fit(X_train, y_train)
        self.model_2_accuracy = clf.score(X_test, y_test)
        pred = clf.predict(X_test)
        self.model_2_hamming_loss = hamming_loss(y_test, pred)

        return

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.

        # Evaluate learned model on testing data, and print the results.
        # print("Accuracy\t" + str(0.0) + "\tHamming loss\t" + str(0.0))
        print("Accuracy\t" + str(self.model_1_accuracy) + "\tHamming loss\t" + str(self.model_1_hamming_loss))
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.

        # Evaluate learned model on testing data, and print the results.
        # print("Accuracy\t" + str(0.0) + "\tHamming loss\t" + str(0.0))
        print("Accuracy\t" + str(self.model_2_accuracy) + "\tHamming loss\t" + str(self.model_2_hamming_loss))
        return
