'''
This script is to:
    load datasets
    choose input groups (e.g. {30 characters (+ G1 (+ G2))} )
    choose output groups according to the input groups
    return training and testing dataset
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(path="data/student-por.csv", predict_kws=['G3']):
    df = pd.read_csv(path, sep=';')

    input_cols = [col for col in df.columns if col not in predict_kws]
    output_cols = predict_kws

    X_df = pd.get_dummies(df[input_cols], drop_first=True)
    Y_df = df[output_cols]

    data_train = []
    y_train = []
    data_test = []
    y_test = []

    for i in range(len(df)):
        x_row = X_df.iloc[i].values
        y_row = Y_df.iloc[i].values

        if i % 5 == 0:
            data_test.append(x_row)
            y_test.append(y_row)
        else:
            data_train.append(x_row)
            y_train.append(y_row)

    scalar = StandardScaler()
    data_train = scalar.fit_transform(data_train)
    data_test = scalar.transform(data_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return data_train, y_train, data_test, y_test


if __name__ == "__main__":
    # test load_data function
    data_train, y_train, data_test, y_test = load_data(predict_kws=['G2', 'G3'])