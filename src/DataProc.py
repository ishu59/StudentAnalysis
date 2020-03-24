# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, chi2
from sklearn.model_selection import GridSearchCV

cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
        'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher', 'internet',
        'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'final_score']
num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc',
            'Walc', 'health', 'absences', 'final_score']
cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'nursery',
            'higher', 'internet', 'romantic', 'edusupport']

bin_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'nursery',
            'higher', 'internet', 'romantic', 'schoolsup', 'familysup', 'paidsup', 'nosup']
multi_cat = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
             'health', 'Fjob', 'reason', 'guardian']
real_cols = ['age', 'absences', 'final_score']
edu_cols = ['schoolsup', 'familysup', 'paidsup', 'nosup']
mjob_dict = {'teacher': '5', 'services': '4', 'health': '3', 'other': '2', 'at_home': '1'}

# A utility method to create a tf.data dataset from a Pandas Dataframe
# def df_to_dataset(dataframe, shuffle=False, batch_size=32, target='target'):
#     dataframe = dataframe.copy()
#     labels = dataframe.pop(target)
#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#     ds = ds.batch(batch_size)
#     return ds


def scale_numeric(train, test, clean_final=False):
    real_cols = ['age', 'absences']
    if clean_final:
        real_cols.append('final_score')
    scalar = MinMaxScaler()
    train[real_cols] = scalar.fit_transform(train[real_cols])
    test[real_cols] = scalar.transform(test[real_cols])
    return train, test


def get_mjob_scores(class_rep):
    mjob_dict = {'teacher': '5', 'services': '4', 'health': '3', 'other': '2', 'at_home': '1'}
    rep_dict = {}
    for k, v in mjob_dict.items():
        rep_dict[k] = class_rep[v]
    return rep_dict




def categorize_mjob(student_data):
    student_data = student_data.copy()
    student_data.loc[student_data['Mjob'] == 'teacher', 'Mjob'] = 5
    student_data.loc[student_data['Mjob'] == 'services', 'Mjob'] = 4
    student_data.loc[student_data['Mjob'] == 'health', 'Mjob'] = 3
    student_data.loc[student_data['Mjob'] == 'other', 'Mjob'] = 2
    student_data.loc[student_data['Mjob'] == 'at_home', 'Mjob'] = 1
    student_data['Mjob'] = student_data['Mjob'].astype(int)
    return student_data


def binarize_student_data(student_data):
    # bin_cols = []
    # for col in student_data:
    #     if len(student_data[col].unique()) <= 2:
    #         bin_cols.append(col)
    # print(bin_cols)
    bin_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'nursery',
                'higher', 'internet', 'romantic']
    # School
    student_data = student_data.copy()
    student_data.loc[student_data['school'] == 'GP', 'school'] = 0
    student_data.loc[student_data['school'] == 'MS', 'school'] = 1
    # Gender
    student_data.loc[student_data['sex'] == 'M', 'sex'] = 0
    student_data.loc[student_data['sex'] == 'F', 'sex'] = 1
    # Address
    student_data.loc[student_data['address'] == 'R', 'address'] = 0
    student_data.loc[student_data['address'] == 'U', 'address'] = 1
    # Family Size
    student_data.loc[student_data['famsize'] == 'LE3', 'famsize'] = 0
    student_data.loc[student_data['famsize'] == 'GT3', 'famsize'] = 1
    # Parent Status
    student_data.loc[student_data['Pstatus'] == 'A', 'Pstatus'] = 0
    student_data.loc[student_data['Pstatus'] == 'T', 'Pstatus'] = 1
    # Nursery
    student_data.loc[student_data['nursery'] == 'yes', 'nursery'] = 1
    student_data.loc[student_data['nursery'] == 'no', 'nursery'] = 0
    # Higher Education
    student_data.loc[student_data['higher'] == 'yes', 'higher'] = 1
    student_data.loc[student_data['higher'] == 'no', 'higher'] = 0
    # Internet
    student_data.loc[student_data['internet'] == 'yes', 'internet'] = 1
    student_data.loc[student_data['internet'] == 'no', 'internet'] = 0
    # Relationship
    student_data.loc[student_data['romantic'] == 'yes', 'romantic'] = 1
    student_data.loc[student_data['romantic'] == 'no', 'romantic'] = 0
    # student_data = student_data.infer_objects()
    return student_data


def multi_categorize_student_data(student_data, clean_mjob):
    # multi_cat = []
    # for col in student_data:
    #     print(col)
    #     print(student_data[col].unique())
    #     if len(student_data[col].unique()) > 2 and len(student_data[col].unique()) < 8:
    #         multi_cat.append(col)
    # print(multi_cat)
    student_data = student_data.copy()
    multi_cat = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
                 'health', 'Fjob', 'reason', 'guardian']
    if clean_mjob:
        multi_cat.append('Mjob')
    df = student_data[multi_cat]
    for col in multi_cat:
        df[col] = df[col].astype(object)
    df = pd.get_dummies(df)
    student_data = student_data.drop(multi_cat, axis=1)
    student_data = pd.concat([student_data, df], axis=1)
    student_data = student_data.infer_objects()
    return student_data


edu_labels = ['schoolsup', 'familysup', 'paidsup', 'nosup']


def clean_edusupport(data):
    data = data.copy()
    data['schoolsup'] = data['edusupport'].apply(lambda x: 1 if 'school' in x else 0)
    data['familysup'] = data['edusupport'].apply(lambda x: 1 if 'family' in x else 0)
    data['paidsup'] = data['edusupport'].apply(lambda x: 1 if 'paid' in x else 0)
    data['nosup'] = data['edusupport'].apply(lambda x: 1 if 'no' in x else 0)
    data = data.drop(columns=['edusupport'])  # , inplace=True
    return data


def read_process_data(data_path, clean_edu=True, clean_mjob=True):
    cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
            'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher', 'internet',
            'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'final_score']
    num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc',
                'Walc', 'health', 'absences', 'final_score']
    cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'nursery',
                'higher', 'internet', 'romantic', 'edusupport']
    data = pd.read_csv(data_path, sep='\t', header=None, names=cols)
    student_data = data.copy()
    student_data = binarize_student_data(student_data.drop('edusupport', axis=1))
    student_data = multi_categorize_student_data(student_data, clean_mjob=clean_mjob)
    edu_data = data[['edusupport']]
    if clean_edu:
        edu_data = clean_edusupport(edu_data.copy())
    student_data = pd.concat([student_data, edu_data], axis=1)
    return student_data


def equalize_test_cols(train, test):
    missing_cols = set(train.columns) - set(test.columns)
    for c in missing_cols:
        test[c] = 0
    rem_cols = set(test.columns) - set(train.columns)
    for c in rem_cols:
        test.drop(c, axis=1)
    test = test[train.columns]
    return train, test


def get_models():
    # model_names = ['LR', 'DTR', 'KNNR', 'SVR', 'ABR', 'RFR']
    models = {}
    alpha = 0.95
    model_names = ['LR', 'DTR', 'KNNR', 'SVR', 'ABR', 'RFR']
    models['LR'] = LinearRegression()
    models['DTR'] = DecisionTreeRegressor()
    models['KNNR'] = KNeighborsRegressor()
    models['SVRL'] = SVR(kernel='linear', C=100, gamma='auto')
    models['SVRR'] = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    models['ABR'] = AdaBoostRegressor(n_estimators=25)
    models['RFR'] = RandomForestRegressor(random_state=1, n_estimators=15)
    models['GBR'] = GradientBoostingRegressor(random_state=1, n_estimators=15)
    return models


def compute_edu_important(X_train, y_train, k, cols=['schoolsup', 'familysup', 'paidsup', 'nosup']):
    df_score = None
    debug = False
    for col in cols:
        k_best = SelectKBest(score_func=chi2, k=k)
        k_best.fit(X_train, y_train[col])
        if df_score is None:
            df_score = pd.DataFrame(k_best.scores_, index=X_train.columns, columns=[col])
        else:
            df = pd.DataFrame(k_best.scores_, index=X_train.columns, columns=[col])
            df_score = pd.concat([df_score, df], axis=1)
    if debug:
        df_score.mean(axis=1).nlargest(k).plot(kind='bar')
    return list(df_score.mean(axis=1).nlargest(k).index)


def compute_important_features(X_train, y_train, classifier=True, k=10, debug=False):
    if classifier:
        k_best = SelectKBest(score_func=chi2, k=k)
    else:
        k_best = SelectKBest(score_func=f_regression, k=k)
    k_best.fit(X_train, y_train)
    df_score = pd.Series(data=k_best.scores_, index=X_train.columns)
    if debug:
        df_score.nlargest(k).plot(kind='bar')
    return list(df_score.nlargest(k).index)


def get_final_score_tts(data, test_data=None, n_best=0):
    if test_data is None:
        X = data.drop('final_score', axis=1)
        Y = data['final_score']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

    else:
        data, test_data = equalize_test_cols(data, test_data)
        X_train = data.drop('final_score', axis=1)
        y_train = data['final_score'].values
        X_test = test_data.drop('final_score', axis=1)
        y_test = test_data['final_score'].values

    if n_best > 0:
        k_best = SelectKBest(score_func=f_regression, k=n_best)
        k_best.fit(X_train, y_train)
        df_score = pd.Series(data=k_best.scores_, index=X_train.columns)
        X_train = X_train[df_score.nlargest(n_best).index]
        print(f'Selected best features are {df_score.nlargest(n_best).index}')
        X_test = X_test[df_score.nlargest(n_best).index]

    return X_train, X_test, y_train, y_test


def trail_main():
    n_folds = 10
    train_path = 'data/assign3_students_train.txt'
    test_path = 'data/assign3_students_test.txt'
    train_data = read_process_data(train_path)
    test_data = read_process_data(test_path)
    models_dict = get_models()
    scores_dict = {}
    learned_models_dict = {}
    for df_key, df_val in train_data.items():
        X_train, X_test, y_train, y_test = get_final_score_tts(df_val.copy(), test_data[df_key].copy(), n_best=15)
        voting_list = []
        for model_key, model_val in models_dict.items():
            model = model_val.fit(X_train, y_train)
            name = f'{df_key}_{model_key}'
            learned_models_dict[name] = model
            voting_list.append((name, model))
            #         print(f"{name}, Train MSE ", mean_squared_error(y_train, model.predict(X_train)))
            #         print(f"{name}, Train RScore ", r2_score(y_train, model.predict(X_train)))
            #         print(f"{name}, Test RScore ", r2_score(y_test, model.predict(X_test)))
            print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            print(f"{name}, Test MSE ", mean_squared_error(y_test, model.predict(X_test)))
            print(f"{name}, Test Score", model.score(X_test, y_test))
            print('=' * 75, '\n')
        model = VotingRegressor(voting_list)
        model = model.fit(X_train, y_train)
        print('=' * 75, '\n')
        print(f"{df_key}, Voting Test MSE = ", mean_squared_error(y_test, model.predict(X_test)))
        print(f"{df_key}, Voting Test Score", model.score(X_test, y_test))
        print('=' * 75, '\n\n')


if __name__ == '__main__':
    print()
    # trail_main()

#
# def clean_cat_student_data_dep(student_data):
#     # School
#     student_data.loc[student_data['school'] == 'GP', 'school'] = 1
#     student_data.loc[student_data['school'] == 'MS', 'school'] = 2
#     # Gender
#     student_data.loc[student_data['sex'] == 'M', 'sex'] = 1
#     student_data.loc[student_data['sex'] == 'F', 'sex'] = 2
#     # Address
#     student_data.loc[student_data['address'] == 'R', 'address'] = 1
#     student_data.loc[student_data['address'] == 'U', 'address'] = 2
#     # Family Size
#     student_data.loc[student_data['famsize'] == 'LE3', 'famsize'] = 1
#     student_data.loc[student_data['famsize'] == 'GT3', 'famsize'] = 2
#     # Parent Status
#     student_data.loc[student_data['Pstatus'] == 'A', 'Pstatus'] = 1
#     student_data.loc[student_data['Pstatus'] == 'T', 'Pstatus'] = 2
#     # Mother's Job Status
#     student_data.loc[student_data['Mjob'] == 'teacher', 'Mjob'] = 5
#     student_data.loc[student_data['Mjob'] == 'services', 'Mjob'] = 4
#     student_data.loc[student_data['Mjob'] == 'health', 'Mjob'] = 3
#     student_data.loc[student_data['Mjob'] == 'other', 'Mjob'] = 2
#     student_data.loc[student_data['Mjob'] == 'at_home', 'Mjob'] = 1
#     # Father's Job Status
#     student_data.loc[student_data['Fjob'] == 'teacher', 'Fjob'] = 5
#     student_data.loc[student_data['Fjob'] == 'services', 'Fjob'] = 4
#     student_data.loc[student_data['Fjob'] == 'health', 'Fjob'] = 3
#     student_data.loc[student_data['Fjob'] == 'other', 'Fjob'] = 2
#     student_data.loc[student_data['Fjob'] == 'at_home', 'Fjob'] = 1
#     # Reasons
#     student_data.loc[student_data['reason'] == 'reputation', 'reason'] = 4
#     student_data.loc[student_data['reason'] == 'course', 'reason'] = 3
#     student_data.loc[student_data['reason'] == 'home', 'reason'] = 2
#     student_data.loc[student_data['reason'] == 'other', 'reason'] = 1
#     # Guardians
#     student_data.loc[student_data['guardian'] == 'father', 'guardian'] = 3
#     student_data.loc[student_data['guardian'] == 'mother', 'guardian'] = 2
#     student_data.loc[student_data['guardian'] == 'other', 'guardian'] = 1
#     # Nursery
#     student_data.loc[student_data['nursery'] == 'yes', 'nursery'] = 2
#     student_data.loc[student_data['nursery'] == 'no', 'nursery'] = 1
#     # Higher Education
#     student_data.loc[student_data['higher'] == 'yes', 'higher'] = 2
#     student_data.loc[student_data['higher'] == 'no', 'higher'] = 1
#     # Internet
#     student_data.loc[student_data['internet'] == 'yes', 'internet'] = 2
#     student_data.loc[student_data['internet'] == 'no', 'internet'] = 1
#     # Relationship
#     student_data.loc[student_data['romantic'] == 'yes', 'romantic'] = 2
#     student_data.loc[student_data['romantic'] == 'no', 'romantic'] = 1
#     student_data = student_data.astype('int32')
#     return student_data
#
#
# def compute_label_encoder_dep(df):
#     enc = LabelEncoder()
#     category_colums = df.select_dtypes('object').columns
#     for i in category_colums:
#         df[i] = enc.fit_transform(df[i])
#     return df
#
#
# # # X_train.to_csv("clean.csv")
# # multi_cat = []
# # lin = []
# # for col in X_train:
# #     print(col)
# #     print(X_train[col].unique())
# #     if len(X_train[col].unique()) > 2 and len(X_train[col].unique()) < 8:
# #         multi_cat.append(col)
# #     if len(X_train[col].unique()) >=8:
# #         lin.append(col)
# # print(multi_cat)
# # print(lin)
#
# def scale_numeric_dep(data, num_cols=None):
#     # for col in num_cols:
#     #     data[col] = scaler.fit_transform(data[col])
#     scaler = MinMaxScaler()
#     data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
#     return data
#
# # def read_process_data_depricated(data_path, raw=False):
# #     cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
# #             'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher', 'internet',
# #             'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'final_score']
# #     num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc',
# #                 'Walc', 'health', 'absences', 'final_score']
# #     cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'nursery',
# #                 'higher', 'internet', 'romantic', 'edusupport']
# #     student_data = pd.read_csv(data_path, sep='\t', header=None, names=cols)
# #     data_frames = {}
# #     # df_names = ['num', 'num_scaled', 'cat_encoded', 'cat_onehot']
# #     student_data_num = student_data.copy()[num_cols]
# #     student_data_cat = student_data.copy()[cat_cols]
# #     student_data_cat_onehot = clean_edusupport(student_data_cat.copy())
# #     data_frames['num'] = student_data.copy()[num_cols]
# #     data_frames['num_scaled'] = scale_numeric(student_data.copy()[num_cols], num_cols)
# #     data_frames['cat_encoded'] = clean_cat_student_data(student_data_cat_onehot.copy())
# #     data_frames['cat_onehot'] = pd.get_dummies(student_data_cat_onehot.copy())
# #
# #     df_learning = {}
# #     df_learning['num_cat_encoded'] = pd.concat([data_frames['num'].copy(), data_frames['cat_encoded'].copy()]
# #                                                , axis=1)
# #     df_learning['num_cat_onehot'] = pd.concat([data_frames['num'].copy(), data_frames['cat_onehot'].copy()]
# #                                               , axis=1)
# #     df_learning['num_scaled_cat_encoded'] = pd.concat(
# #         [data_frames['num_scaled'].copy(), data_frames['cat_encoded'].copy()]
# #         , axis=1)
# #     df_learning['num_scaled_cat_onehot'] = pd.concat(
# #         [data_frames['num_scaled'].copy(), data_frames['cat_onehot'].copy()],
# #         axis=1)
# #     if raw:
# #         df_learning['num_cat'] = pd.concat(
# #             [student_data_num.copy(), student_data_cat_onehot.copy()], axis=1)
# #     return df_learning
#
#
# # def trial_on_raw_dep(train_data, test_data):
# #     train_path = 'data/assign3_students_train.txt'
# #     test_path = 'data/assign3_students_test.txt'
# #     train_data = read_process_data(train_path, raw=True)
# #     test_data = read_process_data(test_path, raw=True)
# #     train = compute_label_encoder(train_data['num_cat'].copy())
# #     test = compute_label_encoder(test_data['num_cat'].copy())
# #     y_train = train['final_score']
# #     X_train = train.drop('final_score', axis=1)
# #     k_best = SelectKBest(score_func=f_regression, k=15)
# #     k_best.fit(X_train, y_train)
# #     df_score = pd.Series(data=k_best.scores_, index=X_train.columns)
# #     # df_score.nlargest(15).index
# #     X_train = X_train[df_score.nlargest(15).index]
