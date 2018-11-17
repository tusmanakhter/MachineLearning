# MachineLearning

Assignment for COMP 472.

Uses scikit-learn to run machine learning algorithms on data to predict letters.

# Requirements
1. Python 3.5+

# Instructions

**Usage**: `python3 model.py [-h] [-d] [-n] [-nn] [-m model_file] [-te test_file] [-tr training_file] [-v validation_file] [-r result_file] [-s save_file]`

To run this program specify the python3 interpreter and the model.py script name with arguments. The program
expects multiple options that can be explained by running the -h or --help command.

The program expects all data files in a folder in the root of this project called DataSet.

The program outputs all result files in a folder in the root of this project called Results.

The program outputs and expects all model files in a folder in the root of this project called Models.

Example:

`python3 model.py -d  -tr 'ds1/ds1Train.csv' -v 'ds1/ds1Val.csv' -r 'ds1/ds1Val-dt.csv' -s 'ds1/ds1Model-dt.pkl'`

The program trains the model using a decision tree and then validates and prints the accuracy.
It uses input file *ds1/ds1Train.csv* for training and *ds1/ds1Val.csv* for validation, both are located in the DataSet folder.
It will output the file *ds1/ds1Val-dt.csv* in folder Results and *ds1/ds1Model-dt.pkl* in folder Models.

`python3 model.py -te 'ds1/ds1Test.csv' -r 'ds1/ds1Test-dt.csv' -m 'ds1/ds1Model-dt.pkl'`

The program tests the model found in the file *ds1/ds1Model-dt.pkl* in the Models folder on the data found in file *ds1/ds1Test.csv* in the DataSet folder.
It will output the file *ds1/ds1Test-dt.csv* in folder Results.

More examples can be found in the `run_testing.sh` and `run_training.sh` files.