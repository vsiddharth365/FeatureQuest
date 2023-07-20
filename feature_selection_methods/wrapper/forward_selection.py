from feature_selection_methods.misc import *

"""
Function to perform feature selection by Forward Selection method
Accepted args: dataframe: pandas dataframe, output_feature_name: string, max_num_of_features_to_select: integer, model: object (sklearn model), scoring: string (scoring method for a model)
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def ForwardSelectionWrapper(dataframe, output_feature_name, max_num_of_features_to_select, model, scoring):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        y = output_dataframe(dataframe, output_feature_name)  # prepare output feature's dataframe
        selected_features = []  # initializing an empty list of selected features
        feature_score = []  # list to store feature scores
        model = model  # set model as the model passed in argument
        best_score = 0.0  # initializing the best score
        # running the loop until 'k' features get selected
        while len(selected_features) < max_num_of_features_to_select:
            best_feature = None  # setting the best feature as null
            """
            for each unselected input feature:
                add it to selected list,
                evaluate the cross validation scores obtained,
                update the best score according to the mean of scores
            """
            for feature in X.columns:
                if feature not in selected_features:
                    current_features = selected_features + [feature]  # add unselected feature to the current features' list
                    X_current = X[current_features]  # filter dataframe for currently selected features
                    scores = cross_val_score(model, X_current, y, scoring=scoring, cv=5, n_jobs=-1)  # calculate cross validation scores for currently selected features
                    mean_score = scores.mean()  # find the mean of cross validation scores
                    if mean_score > best_score:
                        feature_score.append(mean_score - best_score)  # feature score is the difference the feature creates in current the best score
                        best_score = mean_score  # update best score
                        best_feature = feature  # store the best feature
            # if the best score is not increased, the best feature remains null
            if best_feature is None:
                break
            # add the best feature to the selected list
            selected_features.append(best_feature)  # add the best feature to the selected list
        sum_of_scores = sum(feature_score)  # get the sum of all scores
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        for i in range(len(selected_features)):
            selected_feature_importance[selected_features[i]] = (feature_score[i] / sum_of_scores) * 100
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
