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
- [bank_data.csv](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/data/bank_data.csv)
- [churn_library.py](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/churn_library.py) 
- [constants.py](https://github.com/Lamiakas/ChurnPrediction-MLDevops/blob/master/constants.py)
- [churn_script_logging_and_tests.py](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/churn_script_logging_and_tests.py)
- [requirements_py3.6.txt](https://github.com/Lamiaka/ChurnPrediction-MLDevops/blob/master/requirements_py3.6.txt)

The data is stored in the data folder in the `bank_data.csv` file.
The file `churn_library.py` contains the functions used to train and test the models predicting churn on the `bank_data.csv` dataset.
The constants variables are defined in the `constants.py` file.
The `churn_script_logging_and_tests.py` file contains the code testing functions for the `churn_library.py` module.
Finally, the `requirements_py3.6.txt` file contains the list of the packages needed to run the code.

## Running Files
To run the files, you need to first install the packages present in the requirement file using the following command in the terminal:  
`pip install -r requirements_py3.6.txt`

To run the tests and validate that the code will run smoothly, you can run the following command:  
`pytest`

Then, when the tests have run successfully, you can run the following command to train the model on the dataset:  
`python churn_library.py`





