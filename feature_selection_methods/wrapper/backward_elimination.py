from feature_selection_methods.misc import *

"""
Function to perform feature selection by Backward Elimination method
Accepted args: dataframe: pandas dataframe, output_feature_name: string, min_num_of_features_to_select: integer, model: object (sklearn model), scoring: string (scoring method for a model)
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def BackwardEliminationWrapper(dataframe, output_feature_name, min_num_of_features_to_select, model, scoring):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        y = output_dataframe(dataframe, output_feature_name)  # prepare output feature's dataframe
        selected_features = list(X.columns)  # initialize the list of selected features
        model = model  # set model as the model passed in argument
        best_score = cross_val_score(model, X[selected_features], y, scoring=scoring, cv=5, n_jobs=-1).mean()  # set the best score as the mean of cross validation scores
        """
        for each selected feature:
            remove it from selected list,
            evaluate the cross validation scores using currently selected features,
            update the best score according to the mean of scores obtained
        """
        # running the loop until at least 'k' features remain
        while len(selected_features) > min_num_of_features_to_select:
            worst_feature = None  # setting worst feature to null
            for feature in selected_features:
                current_features = selected_features.copy()  # copy selected features in current features
                current_features.remove(feature)  # remove a selected feature
                scores = cross_val_score(model, X[current_features], y, scoring=scoring, cv=5, n_jobs=-1)  # find cross validation scores
                mean_score = scores.mean()  # take the mean of scores obtained
                if mean_score > best_score:  # if mean score is greater than best score, update best score
                    best_score = mean_score
                else:
                    worst_feature = feature  # store the feature to be removed as the worst feature
            # if the best score increases, the worst feature remains null
            if worst_feature is None:
                break
            # remove the worst feature from selected list
            selected_features.remove(worst_feature)
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        for i in range(len(selected_features)):
            selected_feature_importance[selected_features[i]] = 100 / len(list(X.columns))  # all selected features are regarded equally important
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
