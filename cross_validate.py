import random
import numpy as np
import sys
from classification import DecisionTreeClassifier
from eval import Evaluator
from main import read_dataset, get_data
from helpers import most_common, get_max_index, test_on_test


EVAL_METHODS = {
    0: Evaluator.precision,
    1: Evaluator.recall,
    2: Evaluator.f1_score
}


# Shuffles the original dataset and
# partitions into k equally sized subsets
def randomly_split_dataset(dataset, k):
    random.shuffle(dataset)
    f = int(len(dataset) / k)
    subsets = []
    for i in range(k - 1):
        subsets.append(dataset[i:i + f])
    subsets.append(dataset[-f:-1])
    return subsets


# Returns the individual error for a given confusion matrix and evaluation function
def get_error(func, *args):
    array, macro = func(*args)
    return 1 - macro


def get_global_error():

    data_set = read_dataset(sys.argv[1])
    k = int(sys.argv[2])
    # func = EVAL_METHODS.get(sys.argv[3])

    classifier = DecisionTreeClassifier()
    evaluator = Evaluator()

    # Split our original dataset into k random partitions for Cross-Validation
    folds = randomly_split_dataset(data_set, k)
    errors = []
    accuracies = []
    trees = []

    for i in range(k):
        # Extract test set from our k dataset
        test_set = folds[i]

        # Get the remaining sets for training
        training_sets = folds[0:i] + folds[i+1:len(folds)]
        training_set = []
        for set in training_sets:
            for item in set:
                training_set.append(item)

        # Train our Decision Tree Classifier with the training set
        x = np.array([item.attributes for item in training_set])
        y = np.array([item.label for item in training_set])
        classifier.train(x, y)

        trees.append(classifier.tree)

        # Generate predictions and annotations to generate confusion matrix
        predictions = classifier.predict(np.array([item.attributes for item in test_set]))
        annotations = np.array([item.label for item in test_set])

        confusion = evaluator.confusion_matrix(predictions, annotations)

        # Apply a validation function on the confusion matrix
        # to get the error from this test set
        errors.append(get_error(evaluator.f1_score, confusion))
        # errors.append(get_error(func, confusion))

        accuracies.append(evaluator.accuracy(confusion))

    x, y = get_data('test')
    predictions = []
    for tree in trees:
        classifier.tree = tree
        predictions.append(classifier.predict(x))

    final_prediction = []
    for i in range(len(y)):
        labels = [prediction[i] for prediction in predictions]
        label = most_common(labels)
        final_prediction.append(label)

    print("Forest accuracy: {}".format(
        evaluator.accuracy(evaluator.confusion_matrix(final_prediction, y))
    ))

    test_on_test(trees[get_max_index(accuracies)])

    return np.mean(np.array(accuracies)), np.mean(np.array(errors))


if __name__ == "__main__":
    average_accuracy, global_error_estimate = get_global_error()
    print("Average Accuracy: {}\nGlobal error estimate: {}".format(average_accuracy, global_error_estimate))
