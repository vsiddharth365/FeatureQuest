from feature_selection_methods.misc import *

"""
Function to perform feature selection by Mean Absolute Difference filter
Accepted args: dataframe: pandas dataframe, output_feature_name: string, num_of_features_to_select: integer
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def MeanAbsoluteDifferenceFilter(dataframe, output_feature_name, num_of_features_to_select):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        column_means = X.mean()  # find the mean of all input features
        avg_deviations = np.abs(X.sub(column_means, axis='columns')).mean()  # find the average deviations from mean for each feature
        sorted_deviations = avg_deviations.sort_values(ascending=False)  # sort the deviations in descending erder
        sum_of_deviations = sorted_deviations.sum()  # get the sum of sorted deviations
        selected_features = sorted_deviations.index[:num_of_features_to_select].tolist()  # select the features with top 'k' values of deviation
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        for feature in selected_features:
            selected_feature_importance[feature] = (sorted_deviations[feature] / sum_of_deviations) * 100
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
