from feature_selection_methods.misc import *
import pandas as pd

"""
Function to perform feature selection by Dispersion Ratio filter
Accepted args: dataframe: pandas dataframe, output_feature_name: string, num_of_features_to_select: integer
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def DispersionRatioFilter(dataframe, output_feature_name, num_of_features_to_select):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        column_am = X.mean()  # get the arithmetic mean of each feature
        column_gm = np.exp(np.log(X).mean())  # get the geometric mean of each feature
        ratios = column_am / column_gm  # find the dispersion ratio of each feature
        sorted_ratios = ratios.sort_values(ascending=False)  # sort the ratios in descending order
        sum_of_ratios = sorted_ratios.sum()  # find the sum of all ratios
        selected_features = sorted_ratios.index[:num_of_features_to_select].tolist()  # select the features with top 'k' ratios
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        for feature in selected_features:
            percentage = (sorted_ratios[feature] / sum_of_ratios) * 100
            if not pd.isna(percentage):
                selected_feature_importance[feature] = percentage
            else:
                selected_feature_importance[feature] = 0
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
