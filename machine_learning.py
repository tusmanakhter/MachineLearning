#!/usr/bin/python3
from sklearn import tree
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
import os


def extract_data(filename):
    with open(filename, 'r') as file:
        data = [line.split(',') for line in file.read().split('\n')][:-1]
    data = [[int(element) for element in row] for row in data]
    features = [d[:-1] for d in data]
    labels = [d[-1] for d in data]
    return features, labels


def print_results(filename, predicted):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        for i in range(len(predicted)):
            file.write('%d,%d\n' % (i + 1, predicted[i]))


def save_model(filename, classifier):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(classifier, file)


def get_prediction(classifier, features):
    predicted = classifier.predict(features)
    return predicted


def print_accuracy(predicted, labels):
    accuracy = accuracy_score(labels, predicted)
    print(accuracy)


def get_classifier(classifier_type):
    if classifier_type == 'decision':
        return tree.DecisionTreeClassifier(random_state=0)
    elif classifier_type == 'naive':
        return naive_bayes.BernoulliNB()
    elif classifier_type == 'neural':
        return neural_network.MLPClassifier(random_state=0)


def train(training_file, validation_file, results_file, model_file, classifier):
    training_features, training_labels = extract_data('./DataSet/' + training_file)
    classifier.fit(training_features, training_labels)
    validation_features, validation_labels = extract_data('./DataSet/' + validation_file)
    predicted = get_prediction(classifier, validation_features)
    print_accuracy(predicted, validation_labels)
    print_results('./Results/' + results_file, predicted)
    save_model('./Models/' + model_file, classifier)


def run(epochs, classifier_type, training_file, validation_file, results_file, model_file, old_model=None):
    classifier = get_classifier(classifier_type)
    if old_model:
        with open('./Models/' + old_model, 'rb') as file:
            classifier = pickle.load(file)
    for i in range(epochs):
        train(training_file, validation_file, results_file, model_file, classifier)
    # Example to find best hyperparameters
    # criterion_options = ['gini', 'entropy']
    # max_depth_options = [10, 20, 30]
    # param_grid = dict(criterion=criterion_options, max_depth=max_depth_options)
    # grid_search(param_grid, classifier, training_file)


def grid_search(param_grid, classifier, training_file):
    features, labels = extract_data('./DataSet/' + training_file)
    grid = GridSearchCV(classifier, param_grid, cv=10, scoring='accuracy', return_train_score=False)
    grid.fit(features, labels)
    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)


