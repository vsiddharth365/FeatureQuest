from feature_selection_methods.misc import *

"""
Function to perform feature selection by Variance Threshold filter
Accepted args: dataframe: pandas dataframe, output_feature_name: string, num_of_features_to_select: integer
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def VarianceThresholdFilter(dataframe, output_feature_name, num_of_features_to_select):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        variances = X.var()  # calculating the variance of each input feature
        total_variance = sum(variances.tolist())  # find the sum of variances
        top_k_variances = variances.iloc[np.argsort(variances)[-num_of_features_to_select - 1:]]  # selecting the top 'k' variances
        threshold = top_k_variances.min()  # finding the minimum among top 'k' variances
        selected_features = X.columns[(lambda _data: _data.get_support(indices=True))(VarianceThreshold(threshold=threshold).fit(X))].tolist()  # selecting top 'k' features
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        for feature in selected_features:
            selected_feature_importance[feature] = (variances[feature] / total_variance) * 100
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
