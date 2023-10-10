# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we build a machine learning model to predict customer churn. 
We use the Telco Customer Churn dataset from Kaggle. 
The dataset contains 7043 rows and 21 columns. 
Each row represents a customer, each column contains customer’s attributes described 
on the column Metadata. The raw data contains 7043 rows (customers) and 21 columns (features). 
The “Churn” column is our target. The goal of this project is to train a linear and a tree-based model and evaluate their performance on predicting the churn label.
The scripts follow pep8 guidelines and are provided with unit-testing following best coding practices and machine learning DevOps.

## Files and data description
Overview of the files and data present in the root directory. 
The data is stored in the data folder.
The [churn_library.py](http://www.github.com/lamiaka/ChurnPrediction-MLDevops/churn_library.py) file contains the functions used to train and test the model.
It takes in constants variables from the `constants.py` file.
The `churn_script_logging_and_tests.py` file contains the code testing functions.


## Running Files
To run the files, you need to run the following commands in the terminal:  
`pip install -r requirements_py3.6.txt`

Then, you can run the following command to train the model:
`python churn_library.py`

To run the tests, you can run the following command:
`pytest`




