"""
Module storing the constants for churn_library.py and its test module
Author: Lamia
Date: 2023-10-01
"""
# parameters paths
PATH_DATA = './data/bank_data.csv'
IMAGE_FOLDER = './images'
MODEL_FOLDER = './models'
RESULTS_FOLDER = 'results'
EDA_FOLDER = 'eda'
OUTPUT_PATH_RF = './models/rfc_model.pkl'
OUTPUT_PATH_LR = './models/logistic_model.pkl'

# parameters images
FIG_HEIGHT = 10
FIG_WIDTH = 20

# parameters dataset preparation
TEST_SIZE = 0.3
KEEP_COLS_PARAMS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]
QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

# parameters logistic regression training
RANDOM_STATE = 42
MAX_ITER = 3000
SOLVER = 'lbfgs'

# parameters grid search random forest
N_ESTIMATORS_LIST = [200, 500]
MAX_FEATURES_LIST = ['auto', 'sqrt']
MAX_DEPTH_LIST = [4, 5, 100]
CRITERION_LIST = ['gini', 'entropy']
CV = 5
