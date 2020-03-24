from DataProc import read_process_data, categorize_mjob, compute_important_features
from DataProc import num_cols, cat_cols, equalize_test_cols, scale_numeric, get_mjob_scores
# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, accuracy_score
from sklearn.metrics import recall_score, precision_recall_fscore_support, f1_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class Task2:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 2================")
        target = 'Mjob'
        train_path = 'data/assign3_students_train.txt'
        test_path = 'data/assign3_students_test.txt'
        train_data = read_process_data(train_path, clean_edu=True, clean_mjob=False)
        test_data = read_process_data(test_path, clean_edu=True, clean_mjob=False)
        train_data, test_data = scale_numeric(train_data.copy(), test_data.copy())
        train_data, test_data = equalize_test_cols(train_data.copy(), test_data.copy())
        train_data = categorize_mjob(train_data)
        test_data = categorize_mjob(test_data)
        X_train = train_data.drop(target, axis=1)
        X_test = test_data.drop(target, axis=1)
        y_train = train_data[target]
        y_test = test_data[target]
        important_cols = compute_important_features(X_train, y_train, classifier=True, k=35)
        # important_cols
        X_train = X_train[important_cols]
        X_test = X_test[important_cols]

        clf = DecisionTreeClassifier(max_depth=5, criterion="gini")
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rep = classification_report(y_test, pred, output_dict=True)
        self.m1_micro_scores = get_mjob_scores(rep)
        self.m1_acc = accuracy_score(y_test, pred)
        self.m1_prec = precision_score(y_test, pred, average='macro')
        self.m1_recall = recall_score(y_test, pred, average='macro')
        self.m1_f1 = f1_score(y_test, pred, average='macro')

        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rep = classification_report(y_test, pred, output_dict=True)
        self.m2_micro_scores = get_mjob_scores(rep)
        self.m2_acc = accuracy_score(y_test, pred)
        self.m2_prec = precision_score(y_test, pred, average='macro')
        self.m2_recall = recall_score(y_test, pred, average='macro')
        self.m2_f1 = f1_score(y_test, pred, average='macro')

        return

    def print_category_results(self, category, precision, recall, f1):
        print("Category\t" + category + "\tF1\t" + str(f1) + "\tPrecision\t" + str(precision) + "\tRecall\t" + str(
            recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) + "\tMacro_Precision\t" + str(
            precision) + "\tMacro_Recall\t" + str(recall))

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.

        # Evaluate learned model on testing data, and print the results.
        # self.print_macro_results(0.0, 0.0, 0.0, 0.0)
        # categories = ["teacher", "health", "service", "at_home", "other"]
        # for category in categories:
        #     self.print_category_results(category, 0.0, 0.0, 0.0)
        acc = str(round(self.m1_acc, 2))
        prec = str(round(self.m1_prec, 2))
        rec = str(round(self.m1_recall, 2))
        f1 = str(round(self.m1_f1, 2))
        self.print_macro_results(acc, prec, rec, f1)
        categories = ["teacher", "health", "services", "at_home", "other"]
        for category in categories:
            f1 = self.m1_micro_scores[category]['f1-score']
            prec = self.m1_micro_scores[category]['precision']
            rec = self.m1_micro_scores[category]['recall']
            f1 = str(round(f1, 2))
            prec = str(round(prec, 2))
            rec = str(round(rec, 2))
            self.print_category_results(category, f1, prec, rec)

        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.

        # Evaluate learned model on testing data, and print the results.
        # self.print_macro_results(0.0, 0.0, 0.0, 0.0)
        # categories = ["teacher", "health", "services", "at_home", "other"]
        # for category in categories:
        #     self.print_category_results(category, 0.0, 0.0, 0.0)

        acc = str(round(self.m2_acc, 2))
        prec = str(round(self.m2_prec, 2))
        rec = str(round(self.m2_recall, 2))
        f1 = str(round(self.m2_f1, 2))
        self.print_macro_results(acc, prec, rec, f1)
        categories = ["teacher", "health", "services", "at_home", "other"]
        for category in categories:
            f1 = self.m2_micro_scores[category]['f1-score']
            prec = self.m2_micro_scores[category]['precision']
            rec = self.m2_micro_scores[category]['recall']
            f1 = str(round(f1, 2))
            prec = str(round(prec, 2))
            rec = str(round(rec, 2))
            self.print_category_results(category, f1, prec, rec)
        return
