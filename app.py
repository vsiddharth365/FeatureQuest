import copy
import os.path
import re
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, LogisticRegression
from ydata_profiling import ProfileReport
from feature_selection_methods.embedded.lasso_regularization import LassoRegularizationEmbedded
from feature_selection_methods.embedded.random_forest_regressor import RandomForestEmbedded
from feature_selection_methods.filter.chi_square_test import ChiSquareFilter
from feature_selection_methods.filter.correlation_coefficient_filter import CorrelationCoefficientFilter
from feature_selection_methods.filter.dispersion_ratio_filter import DispersionRatioFilter
from feature_selection_methods.filter.fisher_score_filter import FisherScoreFilter
from feature_selection_methods.filter.mean_absolute_difference import MeanAbsoluteDifferenceFilter
from feature_selection_methods.filter.mutual_dependence_filter import MutualDependenceFilter
from feature_selection_methods.filter.variance_threshold_filter import VarianceThresholdFilter
from feature_selection_methods.wrapper.backward_elimination import BackwardEliminationWrapper
from feature_selection_methods.wrapper.exhaustive_selection import ExhaustiveSelection
from feature_selection_methods.wrapper.forward_selection import ForwardSelectionWrapper
import matplotlib.pyplot as plt
from timeseries import plot
import plotly.offline as pyo

app = Flask(__name__)  # object of Flask application
dataset = dataset1 = output_features = numerical_features = ignore_datetime = None  # initialize global variables with None
eda_performed = 0  # set this to 0 since EDA has not been performed
# create a dictionary of available feature selection methods
method_names = {
    "method1": "Variance Thresholding",
    "method2": "Correlation Coefficient",
    "method3": "Dispersion Ratio",
    "method4": "Fisher Score",
    "method5": "Mean Absolute Difference",
    "method6": "Mutual Dependence",
    "method7": "Chi Square Test",
    "method8": "Forward Selection",
    "method9": "Backward Elimination",
    "method10": "Exhaustive Selection",
    "method11": "Lasso Regularization",
    "method12": "Random Forest Regressor"
}


# function to render index page of website
@app.route('/')
def main():
    return render_template("index.html")


# function to render feature selection page
@app.route('/feature_selection', methods=['POST'])
def feature_selection():
    global eda_performed
    eda_performed = 0  # set this to zero when a new file is uploaded
    f = request.files['file']  # get the list of files stored in request class
    global dataset, output_features, numerical_features, ignore_datetime
    if f:  # if file is not null
        f.save(f.filename)  # save the file in current session of flask application
        if f.filename.endswith(".csv"):  # use appropriate pandas methods to read file based on its extension type
            dataset = pd.read_csv(f.filename)
        elif f.filename.endswith(".xlsx"):
            dataset = pd.read_excel(f.filename)
        if dataset is not None:
            if "Unnamed: 0" in dataset.columns:
                dataset = dataset.drop("Unnamed: 0", axis=1)  # drop the redundant "Unnamed: 0" column from dataset
            output_features = dataset.dtypes.to_dict()  # get the dictionary of columns and their data types in the dataset uploaded
            output_features = {column: str(data_type) for column, data_type in output_features.items()}
            # Check if object columns are actually of datetime64[ns] type
            ignore_datetime = []  # list to store the feature names with datetime data type
            # create a wide range of different possible datetime formats which could be present in dataset to trace if a feature recognized as "object" is actually a "datetime" variable
            date_formats = ['%d.%m.%Y|%H:%M:%S', '%m.%d.%Y|%H:%M:%S', '%Y.%m.%d|%H:%M:%S', '%Y.%d.%m|%H:%M:%S', '%m.%Y.%d|%H:%M:%S', '%d.%Y.%m|%H:%M:%S', '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d', '%Y.%d.%m', '%m.%Y.%d', '%d.%Y.%m', '%d-%m-%Y|%H:%M:%S', '%m-%d-%Y|%H:%M:%S', '%Y-%m-%d|%H:%M:%S',
                            '%Y-%d-%m|%H:%M:%S', '%m-%Y-%d|%H:%M:%S', '%d-%Y-%m|%H:%M:%S', '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d', '%Y-%d-%m', '%m-%Y-%d', '%d-%Y-%m', '%d|%m|%Y', '%m|%d|%Y', '%Y|%m|%d', '%Y|%d|%m', '%m|%Y|%d', '%d|%Y|%m', '%d|%m|%Y|%H:%M:%S', '%m|%d|%Y|%H:%M:%S', '%Y|%m|%d|%H:%M:%S',
                            '%Y|%d|%m|%H:%M:%S', '%m|%Y|%d|%H:%M:%S', '%d|%Y|%m|%H:%M:%S', '%d/%m/%Y|%H:%M:%S', '%m/%d/%Y|%H:%M:%S', '%Y/%m/%d|%H:%M:%S', '%Y/%d/%m|%H:%M:%S', '%m/%Y/%d|%H:%M:%S', '%d/%Y/%m|%H:%M:%S', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%Y/%d/%m', '%m/%Y/%d', '%d/%Y/%m']
            for feature, data_type in output_features.items():
                if str(data_type) == "object":  # check the first non-null record in columns with "object" data type
                    first_non_null_record = next((x for x in dataset[feature] if pd.notnull(x)), None)
                    datetime_detected = False
                    if first_non_null_record is not None:
                        for format_ in date_formats:  # iterate over all datetime formats created
                            first_non_null_record = re.sub(r'\s+', '|', str(first_non_null_record))  # if the record contains redundant spaces, remove them
                            if pd.notna(pd.to_datetime(first_non_null_record, errors='coerce', format=format_)):  # if pandas identifies it as a datetime variable
                                dataset[feature] = pd.to_datetime(dataset[feature], errors='coerce', format=format_.replace('|', ' '))  # set the column's datatype to datetime, replacing '|' in date formats with a space ' '
                                output_features[feature] = "datetime64[ns]"  # modify the value of output_features dictionary for this feature
                                ignore_datetime.append(feature)  # add this feature to ignore_datetime list
                                datetime_detected = True
                                break
                    if not datetime_detected:
                        _feature = str(feature).lower().replace(" ", "")
                        if _feature in ["datetime", "date_time", "datetime64", "date_time64", "date", "timestamp", "time_stamp", "time"]:
                            ignore_datetime.append(feature)
                            output_features[feature] = "datetime64[ns]"
                elif str(data_type) == "datetime64[ns]":  # if the feature is already in valid datetime format, add it to the list
                    ignore_datetime.append(feature)
            numerical_features = {}  # create dictionary of numerical data type features, since only their graphs can be plotted
            for feature, data_type in output_features.items():
                if str(data_type) != 'object':
                    numerical_features[feature] = data_type
    return render_template('feature_selection.html', output_features=output_features, ignore_datetime=len(ignore_datetime), numerical_features=numerical_features)


# function to display the selected features
@app.route('/display_features', methods=['POST'])
def display_features():
    global dataset1
    dataset1 = copy.deepcopy(dataset)  # create a deep copy of dataset to preserve the original against further changes
    fs_methods = list(method_names.values())  # get the list of feature selection methods
    method = str(request.form.get('method'))  # get the index of selected method from the request class
    selected_method = method_names[method]  # get the name of selected method
    selected_features = output_feature_name = num_of_features_to_select = None  # set the new variables to None
    if selected_method in fs_methods[:7]:  # if the selected method is of type filter method
        output_feature_name = str(request.form.get('output_feature_name1'))  # take input of the output feature's name and the number of features to select
        ignore_date = request.form.get('ignore_datetime1')  # get the user's choice of ignoring datetime variable(s)
        num_of_features_to_select = int(request.form.get('num_of_features_to_select1'))
        if str(ignore_date) == "yes":  # if user opts to ignore datetime variable, drop the columns with datetime data type
            dataset1 = dataset1.drop(ignore_datetime, axis=1)
    # call the function of selected method from feature_selection_methods package
    if selected_method == fs_methods[0]:
        selected_features = VarianceThresholdFilter(dataset1, output_feature_name, num_of_features_to_select)
    elif selected_method == fs_methods[1]:
        selected_features = CorrelationCoefficientFilter(dataset1, output_feature_name, num_of_features_to_select)
    elif selected_method == fs_methods[2]:
        selected_features = DispersionRatioFilter(dataset1, output_feature_name, num_of_features_to_select)
    elif selected_method == fs_methods[3]:
        selected_features = FisherScoreFilter(dataset1, output_feature_name, num_of_features_to_select)
    elif selected_method == fs_methods[4]:
        selected_features = MeanAbsoluteDifferenceFilter(dataset1, output_feature_name, num_of_features_to_select)
    elif selected_method == fs_methods[5]:
        selected_features = MutualDependenceFilter(dataset1, output_feature_name, num_of_features_to_select)
    elif selected_method == fs_methods[6]:
        selected_features = ChiSquareFilter(dataset1, output_feature_name, num_of_features_to_select)
    elif selected_method == fs_methods[7]:  # if forward selection method is chosen, input the output feature's name, maximum number of features to select, type of model and scoring method to be used
        output_feature_name = str(request.form.get('output_feature_name2'))
        ignore_date = request.form.get('ignore_datetime2')
        if str(ignore_date) == "yes":
            dataset1 = dataset1.drop(ignore_datetime, axis=1)
        max_num_of_features_to_select = int(request.form.get('max_num_of_features_to_select2'))
        model = check_model(str(request.form.get('model2')))
        scoring = str(request.form.get('scoring2'))
        selected_features = ForwardSelectionWrapper(dataset1, output_feature_name, max_num_of_features_to_select, model, scoring)
    elif selected_method == fs_methods[8]:  # if backward elimination method is chosen, input the output feature's name, minimum number of features to select, type of model and scoring method to be used
        output_feature_name = str(request.form.get('output_feature_name3'))
        ignore_date = request.form.get('ignore_datetime3')
        if str(ignore_date) == "yes":
            dataset1 = dataset1.drop(ignore_datetime, axis=1)
        min_num_of_features_to_select = int(request.form.get('min_num_of_features_to_select3'))
        model = check_model(str(request.form.get('model3')))
        scoring = str(request.form.get('scoring3'))
        selected_features = BackwardEliminationWrapper(dataset1, output_feature_name, min_num_of_features_to_select, model, scoring)
    elif selected_method == fs_methods[9]:  # if exhaustive selection is chosen, input the output feature's name, type of model and scoring method to be used
        output_feature_name = str(request.form.get('output_feature_name4'))
        model = check_model(str(request.form.get('model4')))
        scoring = str(request.form.get('scoring4'))
        ignore_date = request.form.get('ignore_datetime4')
        if str(ignore_date) == "yes":
            dataset1 = dataset1.drop(ignore_datetime, axis=1)
        selected_features = ExhaustiveSelection(dataset1, output_feature_name, model, scoring)
    elif selected_method == fs_methods[10]:  # if lasso regularization method is chosen, input the output feature's name and regularization penalty
        output_feature_name = str(request.form.get('output_feature_name5'))
        regularization_penalty = float(request.form.get('penalty5'))
        ignore_date = request.form.get('ignore_datetime5')
        if str(ignore_date) == "yes":
            dataset1 = dataset1.drop(ignore_datetime, axis=1)
        selected_features = LassoRegularizationEmbedded(dataset1, output_feature_name, regularization_penalty)
    elif selected_method == fs_methods[11]:  # if random forest regressor is chosen, input the output feature's name, number of estimators (decision trees) and number of features to select
        output_feature_name = str(request.form.get('output_feature_name6'))
        estimators = int(request.form.get('estimators6'))
        num_of_features_to_select = int(request.form.get('num_of_features_to_select6'))
        ignore_date = request.form.get('ignore_datetime6')
        if str(ignore_date) == "yes":
            dataset1 = dataset1.drop(ignore_datetime, axis=1)
        selected_features = RandomForestEmbedded(dataset1, output_feature_name, estimators, num_of_features_to_select)
    return render_template('selected_features.html', features=selected_features, output_feature=output_feature_name, selected_method=selected_method)


# function to check the type of model selected in wrapper methods
def check_model(model):
    _model = None
    if model == "LinearRegression()":
        _model = LinearRegression()  # LinearRegression() object from sklearn
    elif model == "LogisticRegression()":
        _model = LogisticRegression()  # LogisticRegression() object from sklearn
    return _model


# function to perform EDA of dataset and render its webpage
@app.route('/eda')
def eda():
    global eda_performed
    if not eda_performed:
        profile = ProfileReport(dataset, title="EDA Report")  # generate EDA Report
        profile.to_file("templates/output.html")  # save report at the defined file path
    eda_performed = 1  # set this to 1 since EDA has now been performed
    return render_template('get_eda.html')


# function to render EDA report at the defined route for downloading
@app.route('/output.html')
def report():
    return render_template('output.html')


# function to render graphs webpage
@app.route('/graphs', methods=['POST'])
def graphs():
    global dataset
    first_feature = str(request.form.get('first_feature'))  # get first feature
    second_feature = str(request.form.get('second_feature'))  # get second feature
    dataset_copy = copy.deepcopy(dataset)  # create a deep copy of dataset so that original is not altered by further changes
    dataset_copy = dataset_copy.dropna(subset=[first_feature, second_feature])  # drop the tuples where either first or second feature is null
    figure = scatterPlot(dataset_copy[first_feature], dataset_copy[second_feature], f"Graph of {second_feature} vs {first_feature}", f"{first_feature}", f"{second_feature}")  # get scatter plot
    # figure = plot(dataset[first_feature], dataset[second_feature], f"Graph of {second_feature} vs {first_feature}", f"{first_feature}", f"{second_feature}")
    # figure = pyo.plot(figure, output_type='div', include_plotlyjs=False)
    if figure.startswith("Error: "):
        return render_template('graph.html', x_var=first_feature, y_var=second_feature, figure=figure[7:], error=1)
    return render_template('graph.html', x_var=first_feature, y_var=second_feature, figure=figure, error=0)


# function to obtain scatter plot between two variables
def scatterPlot(x1, y1, label1, x_label, y_label):
    try:
        plt.figure(figsize=(10, 10))  # initialize graph's dimensions
        plt.scatter(x1, y1, label=label1, s=10)  # plot graph
        plt.plot(x1, y1, linestyle='-', marker='', color='blue')  # connect the dots by blue lines
        plt.xlabel(x_label)  # put label for x-axis variable
        plt.ylabel(y_label)  # put label for y-axis variable
        plt.title(f"Scatter Plot of {y_label} vs {x_label}")  # put the title of graph
        plt.legend()  # show the definition of symbols
        plt.xticks(rotation=45)  # rotate the values along x-axis to avoid overlap
        plt.yticks(rotation=45)  # rotate the values along y-axis to avoid overlap
        output_file = f"Scatter Plot of {y_label} vs {x_label}.png"  # define the name of graph for saving
        path = "static/images/"  # define the saving directory
        if not os.path.exists(path):  # if the directory does not exist, create it
            os.mkdir(path)
        path = os.path.join(path, output_file)  # retrieve the path of the saved graph
        plt.savefig(path, dpi=800)  # save the graph at obtained path, with resolution of 800 dots per inch (dpi)
        plt.close()  # close the plot to save memory
        return path
    except Exception as e:
        return f"Error: {e}"


# starting function of flask application
if __name__ == '__main__':
    app.run()
