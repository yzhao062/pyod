'''Utility function for unifying mat files

'''
import os
import h5py
import scipy as sp
import numpy as np

with h5py.File(os.path.join('../datasets', 'http.mat'), 'r') as file:
    print(list(file.keys()))
    X = list(file['X'])
    y = list(file['y'])

X_stack = np.column_stack((X[0], X[1], X[2]))

http = {'X': X_stack,
        'y': y}

sp.io.savemat('http_n.mat', http)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%
import arff


def read_arff(file_path, misplaced_list):
    misplaced = False
    for item in misplaced_list:
        if item in file_path:
            misplaced = True

    file = arff.load(open(file_path))
    data_value = np.asarray(file['data'])
    attributes = file['attributes']

    X = data_value[:, 0:-2]
    if not misplaced:
        y = data_value[:, -1]
    else:
        y = data_value[:, -2]
    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype('float').astype('int').ravel()

    if y.sum() > len(y):
        print(attributes)
        raise ValueError('wrong sum')

    return X, y, attributes


misplaced_list = ['Arrhythmia', 'Cardiotocography', 'Hepatitis', 'ALOI',
                  'KDDCup99']

X, y, attributes = read_arff(os.path.join('../datasets', 'seismic-bumps.arff'),
                             misplaced_list)

num_index = [3, 4, 5, 6, 8, 9, 10, 11, 12, 16]  # 13,14,15 is null
X_num = X[:, num_index].astype('float64')

# %%

# X_stack = np.column_stack((X[0], X[1], X[2]))

seismic = {'X': X_num,
           'y': y}

sp.io.savemat('seismic.mat', seismic)

# %%##########################################################################

from __future__ import division
from __future__ import print_function

import os
import sys
from time import time
import datetime

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import arff


def read_arff(file_path, misplaced_list):
    misplaced = False
    for item in misplaced_list:
        if item in file_path:
            misplaced = True

    file = arff.load(open(file_path))
    data_value = np.asarray(file['data'])
    attributes = file['attributes']

    X = data_value[:, 0:-2]
    if not misplaced:
        y = data_value[:, -1]
    else:
        y = data_value[:, -2]
    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype('float').astype('int').ravel()

    if y.sum() > len(y):
        print(attributes)
        raise ValueError('wrong sum')

    return X, y, attributes


misplaced_list = ['Arrhythmia', 'Cardiotocography', 'Hepatitis', 'ALOI',
                  'KDDCup99']

arff_list = [
    os.path.join('../semantic', 'Annthyroid',
                 'Annthyroid_withoutdupl_07.arff'),
    os.path.join('../semantic', 'Arrhythmia',
                 'Arrhythmia_withoutdupl_46.arff'),
    os.path.join('../semantic', 'Cardiotocography',
                 'Cardiotocography_withoutdupl_22.arff'),
    os.path.join('../semantic', 'HeartDisease',
                 'HeartDisease_withoutdupl_44.arff'),
    os.path.join('../semantic', 'Hepatitis', 'Hepatitis_withoutdupl_16.arff'),
    os.path.join('../semantic', 'InternetAds',
                 'InternetAds_withoutdupl_norm_19.arff'),
    os.path.join('../semantic', 'PageBlocks',
                 'PageBlocks_withoutdupl_09.arff'),
    os.path.join('../semantic', 'Parkinson', 'Parkinson_withoutdupl_75.arff'),
    os.path.join('../semantic', 'Pima', 'Pima_withoutdupl_35.arff'),
    os.path.join('../semantic', 'SpamBase', 'SpamBase_withoutdupl_40.arff'),
    os.path.join('../semantic', 'Stamps', 'Stamps_withoutdupl_09.arff'),
    os.path.join('../semantic', 'Wilt', 'Wilt_withoutdupl_05.arff'),
    #
    os.path.join('../literature', 'ALOI', 'ALOI_withoutdupl.arff'),
    os.path.join('../literature', 'Glass', 'Glass_withoutdupl_norm.arff'),
    os.path.join('../literature', 'Ionosphere',
                 'Ionosphere_withoutdupl_norm.arff'),
    os.path.join('../literature', 'KDDCup99', 'KDDCup99_original.arff'),
    os.path.join('../literature', 'Lymphography',
                 'Lymphography_original.arff'),
    os.path.join('../literature', 'PenDigits',
                 'PenDigits_withoutdupl_norm_v01.arff'),
    os.path.join('../literature', 'Shuttle', 'Shuttle_withoutdupl_v01.arff'),
    os.path.join('../literature', 'Waveform', 'Waveform_withoutdupl_v01.arff'),
    os.path.join('../literature', 'WBC', 'WBC_withoutdupl_v01.arff'),
    os.path.join('../literature', 'WDBC', 'WDBC_withoutdupl_v01.arff'),
    os.path.join('../literature', 'WPBC', 'WPBC_withoutdupl_norm.arff'),
]
from sklearn.ensemble import VotingClassifier
file_names = [
    'Annthyroid',
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',  # too small
    'Hepatitis',  # too small
    'InternetAds',
    'PageBlocks',
    'Parkinson',  # too small
    'Pima',
    'SpamBase',
    'Stamps',
    'Wilt',
    #
    'ALOI',  # too large
    'Glass',  # too small
    'Ionosphere',
    'KDDCup99',  # too large
    'Lymphography',  # data type X contains categorical
    'PenDigits',
    'Shuttle',
    'Waveform',
    'WBC',  # too small
    'WDBC',  # too small
    'WPBC',  # too small
]

assert (len(arff_list) == len(file_names))

for m in range(len(file_names)):
    arff_file = arff_list[m]
    arff_file_name = file_names[m]
    #    print("\n... Processing", arff_file_name, '...')

    X, y, attributes = read_arff(arff_file, misplaced_list)
    print(arff_file_name, X.shape[0], X.shape[1], y.sum(),
          y.sum() / X.shape[0])
