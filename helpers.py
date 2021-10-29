import itertools
import operator

from classification import DecisionTreeClassifier
from eval import Evaluator
from main import get_data


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


def get_max_index(list):
    return helper(list, 0, 0, len(list), 1)


def helper(list, current_index, best_index, list_length, acc):
    if acc < list_length:
        if list[current_index] == max(list[current_index], list[best_index]):
            return helper(list, current_index + 1, current_index, list_length, acc + 1)
        else:
            return helper(list, current_index + 1, best_index, list_length, acc + 1)
    else:
        return best_index


def test_on_test(tree):
    cli = DecisionTreeClassifier()
    cli.is_trained = True
    cli.tree = tree
    evaluator = Evaluator()
    x_test, y_test = get_data('test')

    prediction = cli.predict(x_test)
    confusion = evaluator.confusion_matrix(prediction, y_test)

    accuracy = evaluator.accuracy(confusion)
    print('Accuracy of DT on test.txt: {}'.format(accuracy))
