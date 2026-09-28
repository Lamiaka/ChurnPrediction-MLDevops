# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we identify the credit card customers of a bank who are most likely to churn.
The dataset contains 10,127 customers described by 21 attributes: demographics, card category, credit limit, balances and 12-month transaction activity. The target is `Attrition_Flag` (existing vs. attrited customer).
We train a logistic regression and a random forest, compare them with ROC curves and classification reports, and explain the random forest with feature importance and SHAP values.
The code follows PEP 8, is modular and documented, and comes with unit tests and logging, following software-engineering best practices for machine learning.

## Files and data description
- [data/bank_data.csv](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/data/bank_data.csv): the customer dataset.
- [churn_library.py](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/churn_library.py): functions for EDA, feature engineering, training and evaluating the models.
- [constants.py](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/constants.py): constants shared by the scripts (paths, column lists, parameters).
- [churn_script_logging_and_test.py](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/churn_script_logging_and_test.py): unit tests for `churn_library.py`, with logging.
- [requirements_py3.6.txt](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/requirements_py3.6.txt): the packages needed to run the code.

Generated when the code runs:
- [images](https://github.com/Lamiaka/ChurnPrediction-MLDevops/tree/master/images): EDA plots and model results (ROC curves, classification reports, feature importance, SHAP).
- [models](https://github.com/Lamiaka/ChurnPrediction-MLDevops/tree/master/models): the trained logistic regression and random forest.
- [logs](https://github.com/Lamiaka/ChurnPrediction-MLDevops/tree/master/logs): the test logs.

## Running Files
Install the packages:
`pip install -r requirements_py3.6.txt`

Run the tests:
`pytest`

Generate the test logs:
`python3 churn_script_logging_and_test.py`

Train the models on the dataset:
`python3 churn_library.py`