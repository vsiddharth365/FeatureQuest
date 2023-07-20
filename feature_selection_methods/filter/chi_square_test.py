from feature_selection_methods.misc import *


"""
Function to perform feature selection by chi-square test
Accepted args: dataframe: pandas dataframe, output_feature_name: string, num_of_features_to_select: integer
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def ChiSquareFilter(dataframe, output_feature_name, num_of_features_to_select):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        y = output_dataframe(dataframe, output_feature_name)  # prepare output feature's dataframe
        X_abs = X.apply(lambda x: np.abs(x))  # take absolute values of input features
        y_abs = y.apply(lambda y1: np.abs(y1))  # take absolute values of output feature
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_abs)
        chi2_scores = SelectKBest(score_func=chi2, k=num_of_features_to_select).fit(X_abs, y_encoded).scores_  # find chi2 scores of the 'k' best features
        sum_of_scores = chi2_scores.sum()  # find sum of chi2 scores
        top_k_indices = chi2_scores.argsort()[-num_of_features_to_select:]  # get the features with top 'k' scores
        selected_features = X_abs.columns[top_k_indices].tolist()  # filter the required features
        selected_feature_importance = {}  # dictionary to store selected features and their importance
        for i in range(len(selected_features)):
            selected_feature_importance[selected_features[i]] = (chi2_scores[top_k_indices[i]] / sum_of_scores) * 100
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
