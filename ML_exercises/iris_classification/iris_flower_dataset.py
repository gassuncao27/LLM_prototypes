#!/usr/bin/env python3
from sklearn.datasets import load_iris
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd


def print_princdata(data):
    print(f'\nData Rows and Columns: {data.data.shape}\n')
    print(f'Feature names: {data.feature_names}\n')
    print(f'Data Description: {data.DESCR}')


def transform_df(data):

    array_df = np.column_stack([data.data, data.target])
    df = pd.DataFrame(array_df, columns=[str(data.feature_names[0]),
                                         str(data.feature_names[1]),
                                         str(data.feature_names[2]),
                                         str(data.feature_names[3]),
                                         'Labels'])
    return df


def get_info(data, data_df):
    # nr of observations, missing values, nan values
    nan_values = np.isnan(data.data).sum()
    print(nan_values, type(nan_values), nan_values.shape)
    return data_df.info(), print(f'Nan values: {nan_values}')


def two_scipy(n):
    array_2d = np.eye(n)
    sparse_matrix = csr_matrix(array_2d)
    return array_2d, sparse_matrix


# Program - start
data = load_iris()
df_data = transform_df(data)

print('\nKeys of data: \n----')
for key in data:
    print(key)

# print_princdata(data)
print('\nChecking data info, nan values')
# print(get_info(data), '\n')

array_2d, sparse_matrix = two_scipy(4)
print(sparse_matrix, '\n')

print(df_data.describe(), '\n')
print(df_data['Labels'].unique(), '\n')
print(data.target_names)

labels_name = {'setosa': 0, 'versicolor': 1, 'virginica': 2}


# subamostras - escrever funções
observations = []
for key, values in labels_name.items():
    observations.append(df_data[df_data['Labels'] == values])

print(observations[0])


# Exercise 5 - basic statistics
