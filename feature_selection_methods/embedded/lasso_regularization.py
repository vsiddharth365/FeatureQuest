from feature_selection_methods.misc import *

"""
Function to perform feature selection by Lasso (L1) regularization
Accepted args: dataframe: pandas dataframe, output_feature_name: string, regularization_penalty: float
Returns: selected_feature_importance: dict
selected_feature_importance is a dictionary with key=selected feature, and value=percentage importance of selected feature
"""


def LassoRegularizationEmbedded(dataframe, output_feature_name, regularization_penalty):
    try:
        dataframe = numeric_dataframe(dataframe)  # make all non-numeric features numeric
        X = input_dataframe(dataframe, output_feature_name)  # prepare input features' dataframe
        y = output_dataframe(dataframe, output_feature_name)  # prepare output feature's dataframe
        lasso = Lasso(alpha=regularization_penalty)  # object of Lasso model
        lasso.fit(X, y)  # train the model on X and y
        selected_features = X.columns[lasso.coef_ != 0].tolist()  # select features whose coefficients are non-zero post training
        sum_abs_coeff = sum(abs(lasso.coef_))  # find sum of selected coefficients, larger coefficient indicates more important feature
        selected_feature_importance = {feature: (abs_coeff / sum_abs_coeff) * 100 for feature, abs_coeff in zip(selected_features, abs(lasso.coef_))}  # create dictionary to return selected features and their importance
        return selected_feature_importance
    except Exception as e:
        return {"An error occurred while feature selection": e}
