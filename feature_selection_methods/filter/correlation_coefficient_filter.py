from feature_selection_methods.misc import *

"""
Function to perform feature selection by correlation coefficient filter
Accepted args: dataframe: pandas dataframe, output_feature_name: string, num_of_features_to_select: integer
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def CorrelationCoefficientFilter(dataframe, output_feature_name, num_of_features_to_select):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        y = output_dataframe(dataframe, output_feature_name)  # prepare output feature's dataframe
        correlations = X.apply(lambda x: np.abs(x.corr(y)))  # calculating correlation of each input feature with the output feature
        total_correlation = sum(correlations.tolist())  # get the sum of correlations
        sorted_correlations = correlations.sort_values(ascending=False)  # sorting correlations in descending order
        selected_features = sorted_correlations.index[:num_of_features_to_select].tolist()  # selecting features with top 'k' correlations
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        for feature in selected_features:
            selected_feature_importance[feature] = (correlations[feature] / total_correlation) * 100
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
