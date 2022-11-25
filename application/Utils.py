import json
import ast
import pandas as pd


# Classe que armazenara funcoes comuns

def Utils():
    userCount = 0
    itemCount = 0


def print_timer(start, end):
    diff = end - start
    diff_minute = diff / 60
    diff_hour = diff_minute / 60
    print("Time diff seconds: ", diff)
    print("Time diff minutes: ", diff_minute)
    print("Time diff hours: ", diff_hour)


def analyze(data, dataInfo):
    userCount = data.user_id.unique().shape[0]
    itemCount = data.item_id.unique().shape[0]

    dataInfo["users"] = userCount
    dataInfo["items"] = itemCount

    print("Users: " + str(userCount))
    print("Items: " + str(itemCount))


def build_header(header):
    line = ''
    separator = ';'
    line += 'Techniques'
    for i in range(len(header)):
        line += separator
        line += str(header[i])
    line += '\n'

    return line


def build_file(lst_mse_mae):
    line = ''
    separator = ';'
    file = []

    for key, value in lst_mse_mae.items():
        line += key


        for i in range(len(value)):
            line += separator
            line += str(value[i])

        line += '\n'
        file.append(line)
        line = ''

    return file


def save_file(lst_mse_mae, data_info, k_array, k='average'):
    name_file = data_info["name"]
    name_file += '_'
    name_file += str(k)
    name_file += '.csv'
    path_file = "Results\\Files\\"
    path_file += name_file

    try:
        header = build_header(k_array)
        new_file = build_file(lst_mse_mae)

        with open(path_file, 'w') as f:
            f.writelines(header)
            f.writelines(new_file)

    except Exception as error:
        print(error)


def open_file(name, path, data_info):
    name_file = data_info["name"]
    name_file += '.csv'
    path_file = "Results\\Files\\"
    path_file += name_file

    try:
        dict_final = {}

        df_plot = pd.read_csv(path_file, sep=';')
        k_array = df_plot(index=False).iloc[0:0, 1:]
        df_plot = df_plot.drop(index=df_plot.index[0], axis=0)

        return df_plot

    except Exception as error:
        print(error)
