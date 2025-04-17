'''
This script is to:
    load datasets
    choose input groups (e.g. {30 characters (+ G1 (+ G2))} )
    choose output groups according to the input groups
    return training and testing dataset
'''

import pandas as pd
import numpy as np
import os

def load_data(path="data/student-por.csv", predict_kws=['G3']):
    data = pd.read_csv(path, sep=';')
    data = np.array(data)

    data_train = []
    data_test = []
    y_train = []
    y_test = []

    for person in range(len(data)):
        input_data = [data[person][:-len(predict_kws)]]
        output_data = [data[person][-len(predict_kws):]]

        if person % 5 == 0:
            data_test.append(input_data)
            y_test.append(output_data)
        else:
            data_train.append(input_data)
            y_train.append(output_data)

    return data_train, y_train, data_test, y_test


if __name__ == "__main__":
    # test load_data function
    data_train, y_train, data_test, y_test = load_data(predict_kws=['G2', 'G3'])