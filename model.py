#!/usr/bin/python3
import argparse
import machine_learning


def get_classifier_type(has_decision, has_naive):
    if has_decision and has_naive:
        parser.error("can only run one algorithm at a time")
    elif has_decision:
        needed_type = 'decision'
    elif has_naive:
        needed_type = 'naive'
    return needed_type


parser = argparse.ArgumentParser(description='Trains or tests models')

parser.add_argument("-d", "--decision", help="solve using decision tree", action='store_true')
parser.add_argument("-n", "--naive", help="solve using naive bayes classifier", action='store_true')
parser.add_argument("-m", "--model", help="retrain previously trained model")
parser.add_argument("-te", "--test", help="testing file")
parser.add_argument("-tr", "--training", help="training file")
parser.add_argument("-v", "--validation", help="validation file")
parser.add_argument("-r", "--results", help="results file to save to")
parser.add_argument("-s", "--save", help="model file to save to")
parser.add_argument("-e", "--epochs", help="number of epochs to go through", type=int, default=1)

args = parser.parse_args()
epochs = args.epochs
decision = args.decision
naive = args.naive
model_file = args.model
testing_file = args.test
training_file = args.training
validation_file = args.validation
results_file = args.results
save_model_file = args.save

if testing_file and training_file:
    parser.error("Can only train or test at once, not both")
elif testing_file:
    if not (results_file and model_file and (decision or naive)):
        parser.error("Need testing, results, and model file arguments and an algorithm type")
    else:
        classifier_type = get_classifier_type(decision, naive)
elif training_file:
    if not (validation_file and results_file and save_model_file and (decision or naive)):
        parser.error("Need training, validation, results, and model save file arguments and an algorithm type")
    else:
        classifier_type = get_classifier_type(decision, naive)
        machine_learning.run(epochs, classifier_type, training_file, validation_file, results_file, save_model_file)
else:
    parser.error("Need training or testing file")



