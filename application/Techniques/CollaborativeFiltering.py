import math
from typing import final
from warnings import catch_warnings

import numpy
import numpy as np
import pandas as pd
import sys
import os
import time
import Utils

sys.path.insert(0, 'application/Test/')
from Test import CF_Test
from Techniques import Reports
from scipy import spatial
from numpy import dot, dtype
from numpy.linalg import norm
from sklearn import preprocessing
import scipy.stats


def buildRatingMatrix(db, data_info):
    try:

        ratings = np.zeros((data_info["users"], data_info["items"]), dtype=int)
        for row in db.itertuples():
            ratings[row[1] - 1, row[2] - 1] = row[3]

        return ratings

    except Exception as error:
        print(error)


def calculateSparsity(ratings):
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print("Sparsity: ", sparsity)


def process_similarities(ratings, test_ratings, kind='user'):
    # Inicializa variaveis
    cos = ""
    pearson = ""

    # Processar similaridade apenas de id teste -> todos
    id_test_user, id_test_item = test_ratings.nonzero()
    id_test_user = np.unique(id_test_user)
    id_test_item = np.unique(id_test_item)

    if kind == 'user':
        # Cronometro
        start = time.time()
        print("Start user similarities")

        size = len(id_test_user)
        numRows = ratings.shape[0]

        # Shape matriz de cos e pearson
        cos = np.zeros((numRows, numRows), dtype=np.float32)
        pearson = np.zeros((numRows, numRows), dtype=np.float32)

        for index in range(size):

            # indices de usuario
            i = id_test_user[index]
            any_rating_i = np.any(ratings[i])

            if not any_rating_i:
                continue

            for j in range(numRows):
                try:
                    any_rating_j = np.any(ratings[j])

                    # Verifica se existem valores nos dois vetores
                    if any_rating_i and any_rating_j:

                        sim_cos = dot(ratings[i], ratings[j]) / (norm(ratings[i]) * norm(ratings[j]))
                        sim_pearson, p = scipy.stats.pearsonr(ratings[i], ratings[j])

                    else:
                        sim_cos = 0
                        sim_pearson = 0

                    # Atribuicao para para ixj e jxi
                    cos[i, j] = sim_cos
                    pearson[i, j] = sim_pearson
                    cos[j, i] = sim_cos
                    pearson[j, i] = sim_pearson

                except Exception as error:
                    print("i = {}/{}".format(i + 1, numRows))
                    print("j = {}/{}".format(j + 1, numRows))
                    print(error)

        end = time.time()
        print("Finish user similarities")
        Utils.print_timer(start, end)

    elif kind == 'item':

        # Cronometro
        start = time.time()
        print("Start item similarities")

        size = len(id_test_item)
        numCols = ratings.shape[1]

        # Shape matriz de cos e pearson
        cos = np.zeros((numCols, numCols), dtype=np.float32)
        pearson = np.zeros((numCols, numCols), dtype=np.float32)

        for index in range(size):

            # indices de usuario
            i = id_test_item[index]
            any_rating_i = np.any(ratings[:, i])

            if not any_rating_i:
                continue

            for j in range(numCols):
                try:

                    any_rating_j = np.any(ratings[:, j])

                    # Verifica se existem valores nos dois vetores
                    if any_rating_i and any_rating_j:

                        sim_cos = dot(ratings[:, i], ratings[:, j]) / (norm(ratings[:, i]) * norm(ratings[:, j]))
                        sim_pearson, p = scipy.stats.pearsonr(ratings[:, i], ratings[:, j])

                    else:
                        sim_cos = 0
                        sim_pearson = 0

                    # Atribuicao para para ixj e jxi
                    cos[i, j] = sim_cos
                    pearson[i, j] = sim_pearson
                    cos[j, i] = sim_cos
                    pearson[j, i] = sim_pearson


                except Exception as error:
                    print("i = {}/{}".format(i + 1, numCols))
                    print("j = {}/{}".format(j + 1, numCols))
                    print(error)

        end = time.time()
        print("Finish item similarities")
        Utils.print_timer(start, end)

    return cos, pearson


def predict_topk(ratings, similarity, k, test_ratings, kind='user'):
    pred = np.zeros(ratings.shape)

    # usuarios para testar, itens para testar
    id_test_user, id_test_item = test_ratings.nonzero()
    size = len(id_test_item)

    if kind == 'user':
        # indices de usuario
        for index in range(size):

            # indices de usuario
            i = id_test_user[index]

            # indices de item
            j = id_test_item[index]
            try:
                # Reset

                pred_value = 0
                top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
                any_similarity = np.any(similarity[i, :][top_k_users])
                any_rating = np.any(ratings[:, j][top_k_users])

                if not (any_similarity and any_rating):
                    continue

                average_rating = np.average(ratings[:, j][top_k_users])

                pred_value = similarity[i, :][top_k_users][0].dot(ratings[:, j][top_k_users][0]) \
                             / np.sum(np.abs(similarity[i, :][top_k_users][0]))
                pred[i, j] = pred_value


            except Exception as error:
                print("i = {}/{}".format(i + 1, ratings.shape[0]))
                print("j = {}/{}".format(j + 1, ratings.shape[1]))
                print(error)

    if kind == 'item':
        for index in range(size):

            i = id_test_user[index]
            j = id_test_item[index]

            try:
                # Reset
                pred_value = 0
                top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
                any_similarity = np.any(similarity[j, :][top_k_items])
                any_rating = np.any(ratings[i, :][top_k_items].T)

                if not (any_similarity and any_rating):
                    continue

                average_rating = np.average(ratings[i, :][top_k_items])
                average_rating_t = np.average(ratings[i, :][top_k_items].T)

                pred_value = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred_value /= np.sum(np.abs(similarity[j, :][top_k_items]))
                pred[i, j] = pred_value

                if math.isnan(pred[i, j]):
                    pred[i, j] = 0


            except Exception as error:
                print("i = {}/{}".format(i + 1, ratings.shape[0]))
                print("j = {}/{}".format(j + 1, ratings.shape[1]))
                print(error)

    return pred


def calculate_k(train, test, user_cos_similarity, user_pearson_similarity, item_cos_similarity, item_pearson_similarity,
                k_array):
    user_cos_test_mse = []
    user_cos_test_mae = []

    user_pearson_test_mse = []
    user_pearson_test_mae = []

    item_cos_test_mse = []
    item_cos_test_mae = []

    item_pearson_test_mse = []
    item_pearson_test_mae = []

    list_result = {}

    for k in k_array:
        print("Predict k, k = ", k)
        # ajustar
        user_pred_cos = predict_topk(train, user_cos_similarity, k, test, kind='user')
        user_pred_pearson = predict_topk(train, user_pearson_similarity, k, test, kind='user')
        item_pred_cos = predict_topk(train, item_cos_similarity, k, test, kind='item')
        item_pred_pearson = predict_topk(train, item_pearson_similarity, k, test, kind='item')

        # Calcula MSE and MAE
        user_cos_mse, user_cos_mae = CF_Test.get_MSE_MAE(test, user_pred_cos)
        user_pearson_mse, user_pearson_mae = CF_Test.get_MSE_MAE(test, user_pred_pearson)
        item_cos_mse, item_cos_mae = CF_Test.get_MSE_MAE(test, item_pred_cos)
        item_pearson_mse, item_pearson_mae = CF_Test.get_MSE_MAE(test, item_pred_pearson)

        # Adiciona na lista de user e item
        user_cos_test_mse.append(user_cos_mse)
        user_cos_test_mae.append(user_cos_mae)
        user_pearson_test_mse.append(user_pearson_mse)
        user_pearson_test_mae.append(user_pearson_mae)
        item_cos_test_mse.append(item_cos_mse)
        item_cos_test_mae.append(item_cos_mae)
        item_pearson_test_mse.append(item_pearson_mse)
        item_pearson_test_mae.append(item_pearson_mae)

    list_result["user_cos_test_mse"] = user_cos_test_mse
    list_result["user_pearson_test_mse"] = user_pearson_test_mse
    list_result["user_cos_test_mae"] = user_cos_test_mae
    list_result["user_pearson_test_mae"] = user_pearson_test_mae
    list_result["item_cos_test_mse"] = item_cos_test_mse
    list_result["item_pearson_test_mse"] = item_pearson_test_mse
    list_result["item_cos_test_mae"] = item_cos_test_mae
    list_result["item_pearson_test_mae"] = item_pearson_test_mae

    return list_result


def create_dict_k_neighbors(k_array):
    k_dict = {}

    for item in k_array:
        k_dict[item] = []

    return k_dict


def metric_average_value(lst_dict_metric, k_array):
    lst_dict_average = {}
    lst_dict_std_deviation_under = {}
    lst_dict_std_deviation_above = {}
    lst_dict_sum = {}
    k_dict = create_dict_k_neighbors(k_array)

    # Dicionario de folds
    for key, lst_metric in lst_dict_metric.items():
        lst_k_values = []
        # Dicionario de tecnicas
        for key_2, lst_value in lst_metric.items():
            #Valores dos k vizinhos para cada tecnica
            i = 0
            if key_2 not in lst_dict_sum:
                lst_dict_sum[key_2] = create_dict_k_neighbors(k_array)

            for key_3, sum_value in k_dict.items():
                value = lst_value[i]
                lst_dict_sum[key_2][key_3].append(value)
                i += 1

    for key_tech, lst_values in lst_dict_sum.items():
        current_values_average = []
        current_values_std_deviation = []
        for key_values, values in lst_values.items():
            current_values_average.append(np.average(values))
            current_values_std_deviation.append(np.std(values))
        lst_dict_average[key_tech] = current_values_average
        negative_current_values_std_deviation = np.multiply(current_values_std_deviation, -1)
        lst_dict_std_deviation_under[key_tech] = np.sum([current_values_average, negative_current_values_std_deviation], axis=0)
        lst_dict_std_deviation_above[key_tech] = np.sum([current_values_average, current_values_std_deviation], axis=0)

    return lst_dict_average, lst_dict_std_deviation_under, lst_dict_std_deviation_above


def run(db, data_info):

    try:

        lst_train_test_Fold = CF_Test.build_k_Fold_Train_Test(db)
        k_array = []
        # TO DO descomentar

        #k_array = [5, 10, 20, 50, 75, 100, 150] Exp 1
        k_array = [5, 100, 200, 500, 750, 1000, 1500, 3000]

        lst_result_mse = []
        dict_mse = {}
        count_fold = 1

        for train_test in lst_train_test_Fold:

            # Separando em treino e teste
            db_train = train_test[0]
            db_test = train_test[1]

            # Criando matriz de ratings
            ratings_train = buildRatingMatrix(db_train, data_info)
            ratings_test = buildRatingMatrix(db_test, data_info)
            #calculateSparsity(ratings_train)

            # Processa similaridade
            user_cos_similarity, user_pearson_similarity = process_similarities(ratings_train, ratings_test, kind='user')
            item_cos_similarity, item_pearson_similarity = process_similarities(ratings_train, ratings_test, kind='item')

            # Processa predicao

            result_mse_mae = calculate_k(ratings_train, ratings_test, user_cos_similarity, user_pearson_similarity,
                                         item_cos_similarity, item_pearson_similarity, k_array)
            dict_mse[count_fold] = result_mse_mae

            # Salva resultados
            Utils.save_file(result_mse_mae, data_info, k_array, count_fold)
            count_fold += 1


        average, std_deviation_under, std_deviation_above = metric_average_value(dict_mse, k_array)
        Utils.save_file(average, data_info, k_array, 'average')
        Utils.save_file(std_deviation_under, data_info, k_array, 'standard_deviation')
        Utils.save_file(std_deviation_above, data_info, k_array, 'standard_deviation')
        Reports.plot_CF_MSE(average, k_array, data_info, 'average', std_deviation_under, std_deviation_above)
        Reports.plot_CF_MAE(average, k_array, data_info, 'average', std_deviation_under, std_deviation_above)

        return ''
    except Exception as error:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        print(error)
        print(error.__traceback__)
        print("Exception type: ", exception_type)
        print("File name: ", filename)
        print("Line number: ", line_number)



