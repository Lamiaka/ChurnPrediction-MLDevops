"""
Library of modules for the churn detection project.
Author: Lamia
Date: 2023-10-01
"""

# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import constants as const
#from constants import *
#from constants import image_folder, eda_folder

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(path):
    """
    returns dataframe for the csv found at path

    input:
            path: a path to the csv
    output:
            dataframe: pandas dataframe
    """
    dataframe = pd.read_csv(path)

    return dataframe


def flag_churn(dataframe):
    """
        Create the churn flag based on attrition flag in dataset.
        input:
            dataframe: Pandas DataFrame
        output:
            dataframe: Updated DataFrame with the churn column added
    """
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def perform_eda(dataframe):
    """
        Perform eda on data and save figures to images folder
            input:
                dataframe: input dataframe
                image_folder: folder name where images should be stored

            output:
                None
    """

    # check if eda image path exists
    if not os.path.exists(os.path.join(const.IMAGE_FOLDER, const.EDA_FOLDER)):
        os.makedirs(os.path.join(const.IMAGE_FOLDER, const.EDA_FOLDER))

    # create and save churn distribution
    path_figure = os.path.join(
        const.IMAGE_FOLDER,
        const.EDA_FOLDER,
        'churn_distribution.png')
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=[const.FIG_WIDTH, const.FIG_HEIGHT])

    axis.hist(dataframe['Churn'])
    axis.set_xlabel('churn flag')
    axis.set_ylabel('users')
    fig.savefig(path_figure, format='png')
    plt.close(fig)

    # create and save customer age distribution
    path_figure = os.path.join(
        const.IMAGE_FOLDER,
        const.EDA_FOLDER,
        'customer_age_distribution.png')
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=[const.FIG_WIDTH, const.FIG_HEIGHT])

    axis.hist(dataframe['Customer_Age'])
    axis.set_xlabel('customer age')
    axis.set_ylabel('users')
    fig.savefig(path_figure, format='png')
    plt.close(fig)

    # create and save marital status distribution
    path_figure = os.path.join(
        const.IMAGE_FOLDER,
        const.EDA_FOLDER,
        'marital_status_distribution.png')
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=[const.FIG_WIDTH, const.FIG_HEIGHT])
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar', ax=axis)
    axis.set_xlabel('marital status')
    axis.set_ylabel('user proportion')
    fig.savefig(path_figure, format='png')
    plt.close(fig)

    # create and save total transaction count distribution
    path_figure = os.path.join(
        const.IMAGE_FOLDER,
        const.EDA_FOLDER,
        'total_transaction_count_distribution.png')
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=[const.FIG_WIDTH, const.FIG_HEIGHT])
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True, ax=axis)
    fig.savefig(path_figure, format='png')
    plt.close(fig)

    # create and save feature correlation plot
    path_figure = os.path.join(
        const.IMAGE_FOLDER,
        const.EDA_FOLDER,
        'feature_correlation_plot.png')
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=[const.FIG_WIDTH, const.FIG_HEIGHT])
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2, ax=axis)
    fig.savefig(path_figure, format='png')
    plt.close(fig)


def encoder_helper(dataframe, category_lst, response='Churn'):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                    for naming variables or index y
            column]

    output:
            dataframe: pandas dataframe with new columns for
    """

    for col in category_lst:
        group_lst = []
        groups = dataframe.groupby(col).mean()[response]

        for val in dataframe[col]:
            group_lst.append(groups.loc[val])

        dataframe[col + '_' + response] = group_lst

    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    """
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could be used
                    for naming variables or index y
              column]

    output:
              x_train: training dataset
              x_test:  testing dataset
              y_train: target label training dataset
              y_test: target label testing dataset
    """
    dataframe = encoder_helper(dataframe, category_lst=const.CAT_COLUMNS)
    y_label = dataframe[response]
    x_data = pd.DataFrame()

    keep_cols = const.KEEP_COLS_PARAMS

    x_data[keep_cols] = dataframe[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_label, test_size=const.TEST_SIZE, random_state=const.RANDOM_STATE)
    return x_train, x_test, y_train, y_test


def save_classification_report_image(y_true, y_pred, name):
    """
    Saves the classification report given true and predicted labels.
    input:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels
        name: string describing the subset of labels passed whill will be used
            in the filename of report
    output:
        None
    """
    if not os.path.exists(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER)):
        os.makedirs(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER))

    path_figure = os.path.join(
        const.IMAGE_FOLDER,
        const.RESULTS_FOLDER,
        'classification_report_' +
        name +
        '.png')
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=[const.FIG_WIDTH, const.FIG_HEIGHT])
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, ax=axis)
    fig.savefig(path_figure, format='png')
    plt.close(fig)


def roc_curve_image(x_test, y_test, model_list):
    """
    Saves the plots of ROC curves for all models tested
    input:
        x_test: input features for test dataset
        y_test: response for test dataset
        model_list: list of models to be evaluated
    output:
        None
    """

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=[const.FIG_WIDTH, const.FIG_HEIGHT])
    if not os.path.exists(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER)):
        os.makedirs(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER))

    for model in model_list:
        plot_roc_curve(model, x_test, y_test, ax=axis)
    path_figure = os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER, 'ROC_curves.png')
    fig.savefig(path_figure, format='png')
    plt.close(fig)


def shapley_image(tree_model, x_test):
    """
    Saves the shapley summary plot for the tree based model in a png image
    input:
        tree_model: tree based model such as random forest
        x_test: dataframe with input features for the test dataset
    output:
        None
    """
    if not os.path.exists(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER)):
        os.makedirs(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER))

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    path_figure = os.path.join(
        const.IMAGE_FOLDER,
        const.RESULTS_FOLDER,
        'shapley_image.png')
    plt.savefig(path_figure, format='png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature feature_importance in pth
    input:
            model: model object containing feature_importances_ attribute
            x_data: pandas dataframe of input features
            output_pth: path to store the figure

    output:
             None
    """
    if not os.path.exists(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER)):
        os.makedirs(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER))

    # Calculate feature importance
    feature_importance = model.feature_importances_

    # Sort feature feature_importance in descending order
    indices = np.argsort(feature_importance)[::-1]

    # Rearrange feature names so they match the sorted feature importance
    names = [x_data.columns[i] for i in indices]

    # Create plot
    fig, axis = plt.subplots(figsize=[const.FIG_WIDTH, const.FIG_HEIGHT])

    # Create plot title
    axis.set_title("Feature Importance")
    axis.set_ylabel('Importance')

    # Add bars
    axis.bar(range(x_data.shape[1]), feature_importance[indices])

    # Add feature names as x-axis labels
    axis.set_xticks(range(x_data.shape[1]))
    axis.set_xticklabels(names, rotation=90)
    fig.savefig(output_pth, format='png')
    plt.close(fig)


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: training dataset
              x_test:  testing dataset
              y_train: target label training dataset
              y_test:  target label testing dataset
    output:
              None
    """

    if not os.path.exists(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER)):
        os.makedirs(os.path.join(const.IMAGE_FOLDER, const.RESULTS_FOLDER))

    # grid search
    rfc = RandomForestClassifier(random_state=const.RANDOM_STATE)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(
        solver=const.SOLVER,
        max_iter=const.MAX_ITER,
        random_state=const.RANDOM_STATE)

    param_grid = {
        'n_estimators': const.N_ESTIMATORS_LIST,
        'max_features': const.MAX_FEATURES_LIST,
        'max_depth': const.MAX_DEPTH_LIST,
        'criterion': const.CRITERION_LIST
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=const.CV)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # save classification reports
    save_classification_report_image(y_train, y_train_preds_lr, 'train_lr')
    save_classification_report_image(y_test, y_test_preds_lr, 'test_lr')
    save_classification_report_image(y_train, y_train_preds_rf, 'train_rf')
    save_classification_report_image(y_test, y_test_preds_rf, 'test_rf')

    # save roc curves
    roc_curve_image(x_test, y_test, [lrc, cv_rfc.best_estimator_])

    # save feature importance random forest
    feature_importance_image_path = os.path.join(
        const.IMAGE_FOLDER, const.RESULTS_FOLDER, 'feature_importance.png')
    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_train,
        output_pth=feature_importance_image_path)

    # save shapley image
    shapley_image(cv_rfc.best_estimator_, x_test)

    # save best model
    if not os.path.exists(const.MODEL_FOLDER):
        os.mkdir(const.MODEL_FOLDER)

    joblib.dump(cv_rfc.best_estimator_, const.OUTPUT_PATH_RF)
    joblib.dump(lrc, const.OUTPUT_PATH_LR)


def main(path):
    """
    main function to run the project, saves the exploratory data analysis images,
        trains the models and saves the results
    input:
        path: path to the dataset
    output:
        None
    """
    dataframe = import_data(path)
    dataframe = flag_churn(dataframe)
    perform_eda(dataframe)
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe)
    train_models(x_train, x_test, y_train, y_test)


if __name__ == '__main__':

    main(const.PATH_DATA)
