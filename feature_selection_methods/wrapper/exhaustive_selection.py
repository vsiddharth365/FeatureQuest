from feature_selection_methods.misc import *

"""
Function to perform feature selection by Exhaustive Selection method
Accepted args: dataframe: pandas dataframe, output_feature_name: string, model: object (sklearn model), scoring: string (scoring method for a model)
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def ExhaustiveSelection(dataframe, output_feature_name, model, scoring):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        y = output_dataframe(dataframe, output_feature_name)  # prepare output feature's dataframe
        subsets = []  # initializing an empty list of all subsets of selected features
        feature_score = []  # list to store feature scores
        model = model  # set model as the model passed in argument
        #  Generate all possible combinations of input features to be selected
        for i in range(1, len(X.columns.tolist()) + 1):
            subsets += list(combinations(X.columns.tolist(), i))
        best_score = 0.0  # setting the best score to 0
        best_subset = None  # setting the best subset as null
        # running the loop for each subset of selected features
        for subset in subsets:
            X_subset = X[list(subset)]  # filtering the selected features from the input features' dataframe
            scores = cross_val_score(model, X_subset, y, scoring=scoring, cv=5, n_jobs=-1)  # evaluating the cross validation scores
            mean_score = scores.mean()  # get the mean of feature scores in current subset
            feature_score.append(mean_score)  # store the mean of these scores
            # updating the best score and the best subset according to the mean score
            if mean_score > best_score:  # update best score according to mean score
                best_score = mean_score
                best_subset = subset  # update best subset
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        selected_features = list(best_subset)
        feature_score.sort()  # sort the feature scores of all subsets
        best_feature_importance = np.percentile(feature_score, np.searchsorted(feature_score, best_score) / len(feature_score) * 100)  # find the percentile of best score among all scores
        for feature in selected_features:
            selected_feature_importance[feature] = best_feature_importance
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
