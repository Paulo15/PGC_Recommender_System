import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import sys
import os


def get_MSE_MAE(true_values, pred_values):
    # Ignore nonzero terms.
    true_values_non_zero = true_values.nonzero()
    pred_vector = pred_values[true_values_non_zero].flatten()
    true_vector = true_values[true_values_non_zero].flatten()
    mse = mean_squared_error(y_true=true_vector, y_pred=pred_vector)
    mae = mean_absolute_error(y_true=true_vector, y_pred=pred_vector)

    return mse, mae


def get_data_by_index(train, test, db):
    data_train = db.iloc[train]
    data_test = db.iloc[test]

    return data_train, data_test


def build_k_Fold_Train_Test(db):
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    lst_K_Fold = []
    i = 0

    for train_index, test_index in kf.split(db):
        lst_K_Fold.append(get_data_by_index(train_index, test_index, db))

    return lst_K_Fold
