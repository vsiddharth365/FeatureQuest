from feature_selection_methods.misc import *

"""
Function to perform feature selection by Random Forest Regression
Accepted args: dataframe: pandas dataframe, output_feature_name: string, estimators: integer, num_of_features_to_select: integer
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def RandomForestEmbedded(dataframe, output_feature_name, estimators, num_of_features_to_select):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        y = output_dataframe(dataframe, output_feature_name)  # prepare output feature's dataframe
        rf = RandomForestRegressor(n_estimators=estimators, random_state=42)  # object of Random Forest Regressor model
        rf.fit(X, y)  # train the model
        importance = rf.feature_importances_  # get the list of feature importance
        sorted_indices = importance.argsort()[::-1]  # sort the list in descending order
        selected_features = X.columns[sorted_indices[:num_of_features_to_select]].tolist()  # select the given number of features with the highest importance
        sum_importance = sum(rf.feature_importances_)  # find sum of importance of all features
        selected_feature_importance = {feature: (importance / sum_importance) * 100 for feature, importance in zip(selected_features, importance[sorted_indices[:num_of_features_to_select]])}  # create dictionary to return selected features and their importance
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
