import numpy as np
import copy
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, chi2
from sklearn.feature_selection import VarianceThreshold
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder


def input_dataframe(dataframe, output_feature_name):
    return dataframe.drop([output_feature_name], axis=1)  # prepare input features' dataframe


def output_dataframe(dataframe, output_feature_name):
    return dataframe[output_feature_name]  # prepare output feature's dataframe


def numeric_dataframe(dataframe):
    dataframe = copy.deepcopy(dataframe)  # create a deep copy of dataframe to avoid changes in the original
    non_numeric_cols = dataframe.select_dtypes(exclude="number").columns.tolist()  # get the columns of non-numeric data type
    label_encoder = LabelEncoder()  # object of Label Encoder
    for col in non_numeric_cols:
        dataframe[col] = label_encoder.fit_transform(dataframe[col])  # convert non-numeric columns to numeric
    return dataframe
