"""
Testing module for churn_libray.py module
Author: Lamia
Date: 2023-10-01
"""
# pylint: disable=redefined-outer-name
import os
import shutil
import importlib
import logging
from unittest import mock
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import churn_library as cl


def setup_logging():
    """
    setup basic logging
    """
    logs_directory = './logs'
    logs_filename = 'churn_library.log'
    logs_filepath = os.path.abspath(
        os.path.join(
            logs_directory,
            logs_filename))
    logging.getLogger(__name__)
    logging.basicConfig(
        filename=logs_filepath,
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')


def setup_module():
    """
    setup any state specific to the execution of the given module.
    executed once, before any test in the module is executed.
    """
    setup_logging()


@pytest.fixture(scope="module")
def path():
    """
    fixture for the path to the data
    """
    return cl.const.PATH_DATA


@pytest.fixture(scope="module")
def dataframe():
    """
    fixture for the dataframe before flagging churned customers and feature engineering
    """
    return cl.import_data(cl.const.PATH_DATA)


@pytest.fixture(scope="module")
def dataframe_c():
    """
    fixture for the dataframe after flagging churned customers
    """
    return cl.flag_churn(cl.import_data(cl.const.PATH_DATA))


@pytest.fixture(scope="module")
def category_lst():
    """
    fixture for the category list of features
    """
    return cl.const.CAT_COLUMNS


@pytest.fixture(scope="module")
def dataframe_ce():
    """
    fixture for the dataframe after feature engineering
    """
    return cl.encoder_helper(
        cl.flag_churn(
            cl.import_data(cl.const.PATH_DATA)
        ), cl.const.CAT_COLUMNS
    )


def create_mock_data(n_rows, n_features):
    """
    Create mock data
    input:
        n_rows: input number of rows
        n_features: input number of features
    output:
        x_data: input data
        y_label: input labels
    """
    x_data = pd.DataFrame(columns=[f'feature_{i}' for i in range(10)],
                          data=np.random.randint(0, 10, [n_rows, n_features]))
    y_label = np.random.randint(0, 2, n_rows)
    return x_data, y_label


def create_mock_test_train_split_data(n_rows, n_features, test_size=0.3, random_state=42):
    """
        Create mock data for training the models
        input:
            n_rows: input number of rows
            n_features: input number of features
        output:
            x_train: training data
            x_test: testing data
            y_train: training labels
            y_test: testing labels
        """
    x_data, y_label = create_mock_data(n_rows, n_features)
    x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_label, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def test_import(path):
    """
    test data import - this example is completed for you to assist with the other test functions
    """

    logging.info(10 * '____')
    logging.info('Testing import_data function.')
    logging.info(10 * '____')

    try:
        dataframe = cl.import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_flag_churn(dataframe):
    """
    test flag_churn function
    """
    logging.info(10 * '____')
    logging.info('Testing flag_churn function.')
    logging.info(10 * '____')

    line_count_pre = dataframe.shape[0]
    col_count_pre = dataframe.shape[1]

    logging.info(10 * '____')
    logging.info('Testing flag_churn function.')
    logging.info(10 * '____')
    try:
        dataframe_c = cl.flag_churn(dataframe)
        logging.info('Flagging churned customers: SUCCESS.')
    except AssertionError as err:
        logging.error('Process of flagging churned customers failed.')
        raise err

    try:
        assert dataframe_c.shape[0] == line_count_pre
        assert dataframe_c.shape[1] == col_count_pre + 1
    except AssertionError as err:
        logging.error(
            'The churn column has not been added to the dataframe, '
            'check the churn column assignment process')
        raise err

    try:
        assert dataframe_c['Churn'].unique().tolist() == [0, 1]
    except AssertionError as err:
        logging.error(
            'The churn column values are not 0 or 1, check the churn column assignment process')
        raise err

    return dataframe_c


def test_eda(dataframe_c):
    """
    test perform_eda function
    """
    logging.info(10 * '____')
    logging.info('Testing perform_eda function.')
    logging.info(10 * '____')

    # create mock image folder
    mock_image_folder = './mock_image_folder'
    mock_eda_folder = 'mock_eda_folder'

    if not os.path.exists(mock_image_folder):
        os.mkdir(mock_image_folder)

    if not os.path.exists(os.path.join(mock_image_folder, mock_eda_folder)):
        os.mkdir(os.path.join(mock_image_folder, mock_eda_folder))

    with mock.patch('churn_library.const.IMAGE_FOLDER', mock_image_folder):
        with mock.patch('churn_library.const.EDA_FOLDER', mock_eda_folder):
            # Create a mock object for the fig.savefig function.
            with mock.patch("matplotlib.figure.Figure.savefig") as mock_savefig:
                # Reload the churn_library module to apply the patch
                importlib.reload(cl)
                try:
                    cl.perform_eda(dataframe_c)
                    logging.info('Performing EDA: SUCCESS.')
                except Exception as err:
                    logging.error('Performing EDA: FAILED.')
                    raise err

                try:
                    mock_savefig.assert_any_call(
                        os.path.join(mock_image_folder, mock_eda_folder,
                                     'churn_distribution.png'), format='png')
                    mock_savefig.assert_any_call(
                        os.path.join(
                            mock_image_folder,
                            mock_eda_folder,
                            'customer_age_distribution.png'),
                        format='png')
                    mock_savefig.assert_any_call(
                        os.path.join(
                            mock_image_folder,
                            mock_eda_folder,
                            'marital_status_distribution.png'),
                        format='png')
                    mock_savefig.assert_any_call(
                        os.path.join(
                            mock_image_folder,
                            mock_eda_folder,
                            'total_transaction_count_distribution.png'),
                        format='png')
                    mock_savefig.assert_any_call(
                        os.path.join(
                            mock_image_folder,
                            mock_eda_folder,
                            'feature_correlation_plot.png'),
                        format='png')
                    logging.info('Figure saving for EDA process: SUCCESS.')
                except AssertionError as err:
                    logging.error('Some figures did not get saved.')
                    raise err

    if os.path.exists(mock_image_folder):
        shutil.rmtree(mock_image_folder)


def test_encoder_helper(dataframe_c, category_lst):
    """
    test encoder helper
    """
    logging.info(10 * '____')
    logging.info('Testing encoder_helper function.')
    logging.info(10 * '____')

    try:
        assert isinstance(category_lst, list)
        logging.info('Passing a list of features: SUCCESS')
    except AssertionError as err:
        logging.error('The argument passed was not a list.')
        raise err

    try:
        dataframe_ce = cl.encoder_helper(dataframe_c, category_lst)

        for category in category_lst:
            assert isinstance(category, str)
            assert category + '_Churn' in dataframe_ce.columns

        logging.info('Churn columns created: SUCCESS')

    except AssertionError as err:
        logging.error('The churn columns failed to be created.')
        raise err

    try:
        assert isinstance(dataframe_ce, pd.DataFrame)
    except AssertionError as err:
        logging.error('data returned is not a dataframe')
        raise err


def test_perform_feature_engineering(dataframe_ce):
    """
    test feature engineering process
    """
    logging.info(10 * '____')
    logging.info('Testing perform_feature_engineering function.')
    logging.info(10 * '____')

    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            dataframe_ce)
        logging.info('Feature engineering completed.')

    except Exception as err:
        logging.error('Feature engineering failed.')
        logging.error(err)
        raise err

    try:
        assert len(y_train) + len(y_test) == dataframe_ce.shape[0]
        assert x_train.shape[1] == len(cl.const.KEEP_COLS_PARAMS)
        assert x_test.shape[1] == len(cl.const.KEEP_COLS_PARAMS)
        logging.info('Shape of train and test datasets match input data.')

    except AssertionError as err:
        logging.error(
            'Seems like the datasets do not have the correct shape, '
            'check the feature engineering process.')
        raise err

    try:
        assert list(set(y_train)) == [0, 1]
        assert list(set(y_test)) == [0, 1]
        logging.info('Target labels correctly generated.')

    except AssertionError as err:
        logging.error('Target label is not in [0,1], check target label.')
        raise err


def test_save_classification_report_image():
    """
    test the function which generates the classification report image
    """

    logging.info(10 * '____')
    logging.info('Testing save_classification_report_image function.')
    logging.info(10 * '____')

    # creating fake true and predicted labels:

    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    name = 'test'

    # create mock image folder
    mock_image_folder = './mock_image_folder'
    mock_results_folder = 'mock_results_folder'

    if not os.path.exists(mock_image_folder):
        os.mkdir(mock_image_folder)

    if not os.path.exists(
        os.path.join(
            mock_image_folder,
            mock_results_folder)):
        os.mkdir(os.path.join(mock_image_folder, mock_results_folder))

    with mock.patch('churn_library.const.IMAGE_FOLDER', mock_image_folder):
        with mock.patch('churn_library.const.RESULTS_FOLDER', mock_results_folder):
            # Create a mock object for the fig.savefig function.
            with mock.patch("matplotlib.figure.Figure.savefig") as mock_savefig:
                # Reload the churn_library module to apply the patch
                importlib.reload(cl)
                try:
                    cl.save_classification_report_image(y_true, y_pred, name)
                    path_figure = os.path.join(
                        mock_image_folder,
                        mock_results_folder,
                        'classification_report_' + name + '.png')
                    mock_savefig.assert_any_call(path_figure, format='png')
                    logging.info(
                        'Classification report figure generation: SUCCESS.')
                except AssertionError as err:
                    logging.error(
                        'Classification report figure generation failed.')
                    raise err

    if os.path.exists(mock_image_folder):
        shutil.rmtree(mock_image_folder)


def test_roc_curve_image():
    """
    test the plots of ROC curves for all models tested and saving the images
    """

    logging.info(10 * '____')
    logging.info('Testing roc_curve_image function.')
    logging.info(10 * '____')

    # creating fake dataset and model:
    x_test, y_test = create_mock_data(100, 10)
    model = LogisticRegression()
    model.fit(x_test, y_test)

    mock_image_folder, mock_results_folder = './mock_image_folder', 'mock_results_folder'
    os.makedirs(os.path.join(mock_image_folder, mock_results_folder))

    with mock.patch('churn_library.const.IMAGE_FOLDER', mock_image_folder):
        with mock.patch('churn_library.const.RESULTS_FOLDER', mock_results_folder):
            # Create a mock object for the fig.savefig function.
            with mock.patch("matplotlib.figure.Figure.savefig") as mock_savefig:
                # Reload the churn_library module to apply the patch
                importlib.reload(cl)
                try:
                    cl.roc_curve_image(x_test, y_test, [model])
                    path_figure = os.path.join(
                        mock_image_folder, mock_results_folder, 'ROC_curves.png')
                    mock_savefig.assert_any_call(path_figure, format='png')
                    logging.info('ROC curve figure generation: SUCCESS.')
                except AssertionError as err:
                    logging.error('ROC curve figure generation failed.')
                    raise err

    if os.path.exists(mock_image_folder):
        shutil.rmtree(mock_image_folder)


def test_shapley_image():
    """
    test the plots of shapley values and saving the image
    """

    logging.info(10 * '____')
    logging.info('Testing shapley_image function.')
    logging.info(10 * '____')

    # creating fake dataset and model:
    x_test, y_test = create_mock_data(100, 10)
    model = RandomForestClassifier()
    model.fit(x_test, y_test)

    mock_image_folder = './mock_image_folder'
    mock_results_folder = 'mock_results_folder'

    if not os.path.exists(mock_image_folder):
        os.mkdir(mock_image_folder)

    if not os.path.exists(
        os.path.join(
            mock_image_folder,
            mock_results_folder)):
        os.mkdir(os.path.join(mock_image_folder, mock_results_folder))

    with mock.patch('churn_library.const.IMAGE_FOLDER', mock_image_folder):
        with mock.patch('churn_library.const.RESULTS_FOLDER', mock_results_folder):
            # Create a mock object for the plt.savefig function.
            with mock.patch("matplotlib.pyplot.savefig") as mock_savefig:
                # Reload the churn_library module to apply the patch
                importlib.reload(cl)
                try:
                    cl.shapley_image(model, x_test)
                    path_figure = os.path.join(
                        mock_image_folder, mock_results_folder, 'shapley_image.png')
                    mock_savefig.assert_any_call(path_figure, format='png')
                    logging.info('Shapley figure generation: SUCCESS.')
                except AssertionError as err:
                    logging.error('Shapley figure generation failed.')
                    raise err

    if os.path.exists(mock_image_folder):
        shutil.rmtree(mock_image_folder)


def test_feature_importance_plot():
    """
    test feature_importance_plot function
    """
    logging.info(10 * '____')
    logging.info('Testing feature_importance_plot function.')
    logging.info(10 * '____')

    # creating fake dataset and model:
    x_data, y_data = create_mock_data(100, 10)
    model = RandomForestClassifier()
    model.fit(x_data, y_data)
    output_pth = './test.png'

    try:
        with mock.patch("matplotlib.figure.Figure.savefig") as mock_savefig:

            cl.feature_importance_plot(model, x_data, output_pth)
            mock_savefig.assert_called_once_with(output_pth, format='png')
            logging.info('Feature importance plot : SUCCESS.')
    except AssertionError as err:
        logging.error('Feature importance plot failed.')
        raise err


def test_train_models_model_creation():
    """
    create a test function for train_models function
    that checks that the model training has happened
    """

    logging.info(10 * '____')
    logging.info('Testing train_models function - model training and storage.')
    logging.info(10 * '____')

    # Define some mock data for testing
    x_train, x_test, y_train, y_test = create_mock_test_train_split_data(100, 10)

    # Define some mock paths for model storage
    mock_random_forest_model_path = './models/mock_output_rf.pkl'
    mock_logistic_regression_model_path = './models/mock_output_lr.pkl'

    # Use the patch method to temporarily replace the constants with mock
    # values
    with mock.patch('churn_library.const.OUTPUT_PATH_RF', mock_random_forest_model_path):
        with mock.patch('churn_library.const.OUTPUT_PATH_LR', mock_logistic_regression_model_path):
            try:
                # Reload the churn_library module to apply the patch
                importlib.reload(cl)
                # Training the models
                cl.train_models(x_train, x_test, y_train, y_test)
                logging.info('Training of models: SUCCESS.')
            except Exception as err:
                logging.info('Training of models: FAILED.')
                raise err

            try:
                assert os.path.exists(mock_random_forest_model_path)
                logging.info(
                    'Random forest model path verification: SUCCESS.')
            except AssertionError as err:
                logging.error(
                    'Random forest model path verification: FAILED.')
                raise err

            try:
                assert os.path.exists(
                    mock_logistic_regression_model_path)
                logging.info(
                    'Logistic regression model path verification: SUCCESS.')
            except AssertionError as err:
                logging.error(
                    'Logistic regression model path verification: FAILED.')
                raise err

    os.remove(mock_random_forest_model_path)
    os.remove(mock_logistic_regression_model_path)


def test_train_models_figure_creation():
    """
    create a test function for train_models function that checks
    that the figures have been created
    """
    logging.info(10 * '____')
    logging.info('Testing train_models function - figure creation.')
    logging.info(10 * '____')

    # Define some mock data for testing
    x_train, x_test, y_train, y_test = create_mock_test_train_split_data(100, 10)

    # Define some mock paths for testing
    mock_image_folder, mock_results_folder = './mock_image_folder', 'mock_results_folder'
    mock_random_forest_model_path, mock_logistic_regression_model_path = \
        './models/mock_output_rf.pkl', './models/mock_output_lr.pkl'

    # Create the mock image folder and paths for results
    os.makedirs(os.path.join(mock_image_folder, mock_results_folder))

    with mock.patch('churn_library.const.IMAGE_FOLDER', mock_image_folder):
        with mock.patch('churn_library.const.RESULTS_FOLDER', mock_results_folder):
            with mock.patch('churn_library.const.OUTPUT_PATH_RF', mock_random_forest_model_path):
                with mock.patch('churn_library.const.OUTPUT_PATH_LR',
                                mock_logistic_regression_model_path):
                    # Reload the churn_library module to apply the patch
                    importlib.reload(cl)
                    try:
                        # Training the models
                        cl.train_models(x_train, x_test, y_train, y_test)
                        logging.info('Training of models: SUCCESS.')
                    except Exception as err:
                        raise err

                    # Define the expected mock path for the figures
                    expected_root_path = os.path.join(mock_image_folder, mock_results_folder)

                    # Assert that the paths are constructed correctly
                    try:
                        assert os.path.exists(os.path.join(expected_root_path, 'ROC_curves.png'))
                        logging.info(
                            'ROC curve figure path verification: SUCCESS.')
                    except AssertionError as err:
                        logging.error(
                            'ROC curve figure path verification: FAILED.')
                        raise err

                    try:
                        assert os.path.exists(os.path.join(expected_root_path,
                                                           'classification_report_train_rf.png'))
                        logging.info(
                            'Classification report for random forest figure '
                            'paths verification: SUCCESS.')
                    except AssertionError as err:
                        logging.error(
                            'Classification report for random forest figure '
                            'paths verification: FAILED.')
                        raise err

                    try:
                        assert os.path.exists(os.path.join(expected_root_path,
                                                           'classification_report_train_lr.png'))
                        logging.info(
                            'Classification report for logistic regression figure '
                            'paths verification: SUCCESS.')
                    except AssertionError as err:
                        logging.error(
                            'Classification report for logistic regression figure '
                            'paths verification: FAILED.')
                        raise err

                    try:
                        assert os.path.exists(os.path.join(expected_root_path,
                                                           'feature_importance.png'))
                        logging.info(
                            'Feature importance plot figure '
                            'paths verification: SUCCESS.')
                    except AssertionError as err:
                        logging.error(
                            'Feature importance plot figure '
                            'paths verification: FAILED.')
                        raise err

                    try:
                        assert os.path.exists(os.path.join(expected_root_path,
                                                           'shapley_image.png'))
                        logging.info(
                            'Shapley image figure paths verification: SUCCESS.')
                    except AssertionError as err:
                        logging.error(
                            'Shapley image figure paths verification: FAILED.')
                        raise err

    shutil.rmtree(mock_image_folder)


if __name__ == "__main__":

    setup_logging()
    test_import(cl.const.PATH_DATA)
    test_flag_churn(cl.import_data(cl.const.PATH_DATA))
    test_eda(cl.flag_churn(cl.import_data(cl.const.PATH_DATA)))
    test_encoder_helper(cl.flag_churn(cl.import_data(cl.const.PATH_DATA)), cl.const.CAT_COLUMNS)
    test_perform_feature_engineering(
        cl.encoder_helper(
            cl.flag_churn(cl.import_data(cl.const.PATH_DATA)),
            cl.const.CAT_COLUMNS
        )
    )
    test_save_classification_report_image()
    test_roc_curve_image()
    test_shapley_image()
    test_feature_importance_plot()
    test_train_models_model_creation()
    test_train_models_figure_creation()

