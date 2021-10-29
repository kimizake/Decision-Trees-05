from classification import DecisionTreeClassifier
from data_item import data_item
import sys
import numpy as np
from eval import Evaluator


# File passed in as a parameter - ref via sys.argv[1] (0 refers to file name)
def read_dataset(name):
    dataset = []
    file = open('./data/{}.txt'.format(name))
    for line in file:
        line = line.strip('\n').strip().split(',')
        item = data_item(line)
        dataset.append(item)
    return dataset


def get_data(data):
    dataset = read_dataset(data)
    x = np.array([item.attributes for item in dataset])
    y = np.array([item.label for item in dataset])
    return x, y


if __name__ == "__main__":
    x, y = get_data(sys.argv[1])

    classifier = DecisionTreeClassifier()
    classifier = classifier.train(x, y)

    test_data = read_dataset(sys.argv[2])
    x_test = np.array([item.attributes for item in test_data])
    predictions = classifier.predict(x_test)

    annotations = np.array([item.label for item in test_data])

    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, annotations)

    print("Confusion matrix:")
    print(confusion)

    print("Accuracy:")
    print(evaluator.accuracy(confusion))

    print("Precision:")
    print(evaluator.precision(confusion))

    print("Recall:")
    print(evaluator.recall(confusion))

    print("F1 Measure:")
    print(evaluator.f1_score(confusion))

