import numpy as np
import pandas as pd
import uuid
import time
import random


def generate_id(old_id, db, column_name):
    size = len(old_id)
    id = {}
    i = 0
    start = time.time()
    print("Start generate id: ", start)

    for register in old_id:

        id[register] = i+1
        i += 1

    db[column_name] = db[column_name].replace(id)
    end = time.time()
    diff = end - start
    print("Finish generate id: ", end)
    print("Time diff seconds: ", diff)
    print("Time diff minutes: ", diff/60)

    return id, db




def base_treatment(db):

    distinct_item = db['item_id'].unique()
    distinct_user = db['user_id'].unique()
    item_id, db = generate_id(distinct_item, db, 'item_id')
    user_id, db = generate_id(distinct_user, db, 'user_id')

    return db

def base_treatment(db, number_user = None, number_item = None):

    # Define semente do random
    random.seed(2)

    # Identifica ids
    distinct_item = db['item_id'].unique()
    distinct_user = db['user_id'].unique()

    if not number_user == None:

        # Seleciona usuarios por parâmetro da funcao
        distinct_user = random.choices(distinct_user, k=number_user)

        # Seleciona todos registros que contenham os usuários da lista
        db = db[db['user_id'].isin(distinct_user)]
        db.sort_values(by=['user_id'])

    elif not number_item == None:
        # Seleciona usuarios por parâmetro da funcao
        distinct_user = random.choices(distinct_user, k=number_user)

        # Seleciona todos registros que contenham os usuários da lista
        db = db[db['item_id'].isin(distinct_user)]
        db.sort_values(by=['item_id'])

    distinct_item = db['item_id'].unique()
    distinct_user = db['user_id'].unique()

    item_id, db = generate_id(distinct_item, db, 'item_id')
    user_id, db = generate_id(distinct_user, db, 'user_id')

    return db


def SearchData(dataInfo):

    if(dataInfo["id"] == 0): #MovieLens
        
        columns = ['user_id', 'item_id', 'rating', 'timestamp']

        data = pd.read_csv(dataInfo["path"], sep='\t', names=columns)
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
       

    elif(dataInfo["id"] == 1): #Amazon - Valores entre 2009 e 2012

        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_csv(dataInfo["path"], sep=',', names=columns)
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
        data.hist()
        # Amostra utilizada no PGC
        data = data[(data.datetime >= '2009-01-01') & (data.datetime <= '2012-01-01')]
        data = base_treatment(data)

    elif (dataInfo["id"] == 2):  # Book Rating

        columns = ['user_id', 'item_id', 'rating']
        data = pd.read_csv(dataInfo["path"], sep=';', names=columns, encoding='unicode_escape', low_memory=False)

        # Remove header
        data.drop(index=data.index[0], axis=0, inplace=True)

        # Remove ratings de valor 0
        data = data[data['rating'] != 0]
        data = data[data['rating'] != str(0)]
        data = data[0:10000]
        # Seleciona amostra e transforma ids
        data = base_treatment(data)

    return data

    


