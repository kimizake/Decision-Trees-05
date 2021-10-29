import copy
import sys

from classification import DecisionTreeClassifier, majority_label
from eval import Evaluator
from graphical_tree_visualisation import Visualiser
from main import get_data
from node import leaf_node, split_node
from helpers import test_on_test


class BasicPrune(object):

    def __init__(self, dataset):
        x, y = get_data(dataset)
        self.original_tree = DecisionTreeClassifier().train(x, y).tree

    def run(self):
        self.pruned_tree = self.prune(self.original_tree, get_max_tree_depth(self.original_tree))
        self.visualiser = Visualiser(tree_root=self.pruned_tree, max_depth=50)

    def draw(self):
        self.visualiser.draw()

    def prune(self, tree, depth):
        if depth < 0:
            raise Exception
        if depth == 0:
            return tree

        _tree = copy.deepcopy(tree)

        nodes = get_split_nodes(_tree, depth)

        for parent, child in nodes:
            _tree = replace(_tree, parent, child)
            tree = compare_accuracy(tree, _tree)

        return self.prune(tree, depth-1)


# Return max tree depth relative to subtree.
# Calling with tree root returns depth of whole tree.
def get_max_tree_depth(tree):
    if isinstance(tree, leaf_node):
        return 0
    elif isinstance(tree, split_node):
        return max(
            1 + get_max_tree_depth(tree.left),
            1 + get_max_tree_depth(tree.right)
        )


# Returns list of all the split nodes at given depth level which have two leaf node children
def get_split_nodes(tree, depth):
    return _get_split_nodes(None, tree, depth, 0)


def _get_split_nodes(parent_node, node, depth, acc):
    if acc == depth:
        if isinstance(node, split_node) and isinstance(node.left, leaf_node) and isinstance(node.right, leaf_node):
            return [(parent_node, node)]
        else:
            return []
    elif acc < depth:
        if isinstance(node, split_node):
            return _get_split_nodes(node, node.left, depth, acc+1) + _get_split_nodes(node, node.right, depth, acc+1)
        else:
            return []
    else:
        raise Exception


# Given a prune node (child), we replace it with the majority vote leaf node
def replace(tree, parent, child):
    leaf = leaf_node(majority_label(child.dataset))
    if parent.left == child:
        parent.left = leaf
    else:
        parent.right = leaf
    return tree


# Returns tree with higher accuracy
def compare_accuracy(tree, _tree):
    evaluator = Evaluator()
    x, y = get_data('validation')

    cli = DecisionTreeClassifier()
    cli.is_trained = True
    cli.tree = tree
    p = cli.predict(x)
    c = evaluator.confusion_matrix(p, y)
    a = evaluator.accuracy(c)

    _cli = DecisionTreeClassifier()
    _cli.is_trained = True
    _cli.tree = _tree
    _p = _cli.predict(x)
    _c = evaluator.confusion_matrix(_p, y)
    _a = evaluator.accuracy(_c)

    return _tree if _a > a else tree


if __name__ == '__main__':
    prune = BasicPrune(sys.argv[1])
    prune.run()
    prune.draw()
    test_on_test(prune.pruned_tree)
