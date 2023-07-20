from feature_selection_methods.misc import *

"""
Function to perform feature selection by Mutual Dependence filter
Accepted args: dataframe: pandas dataframe, output_feature_name: string, num_of_features_to_select: integer
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def MutualDependenceFilter(dataframe, output_feature_name, num_of_features_to_select):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        y = output_dataframe(dataframe, output_feature_name)  # prepare output feature's dataframe
        mutual_dependence_scores = SelectKBest(score_func=mutual_info_regression, k=num_of_features_to_select).fit(X, y).scores_  # get the mutual dependence scores for all input features
        sum_of_scores = mutual_dependence_scores.sum()  # find the sum of mutual dependence score
        top_k_indices = mutual_dependence_scores.argsort()[-num_of_features_to_select:]  # get indices of 'k' features with the highest score
        selected_features = X.columns[top_k_indices].tolist()  # filter the selected features by obtained indices
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        for i in range(len(selected_features)):
            selected_feature_importance[selected_features[i]] = (mutual_dependence_scores[top_k_indices[i]] / sum_of_scores) * 100
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
