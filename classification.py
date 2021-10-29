##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np
from data_item import data_item
from node import *
import math
import pickle
import os
import sys
from collections import Counter
from text_tree_visualisation import print_node


def set_entropy(dataset):
    # count occurrences of each ground truth
    occurrences = {}
    for item in dataset:
        if item.label in occurrences:
            occurrences[item.label] += 1
        else:
            occurrences[item.label] = 1
    dataset_size = len(dataset)
    entropy = 0
    for (label, count) in occurrences.items():
        probability = count / dataset_size
        entropy -= probability * math.log(probability, 2)
    return entropy


def potential_split_points(dataset):
    # Find all split points by ordering values in each feature and considering adjacent values with different labels
    split_points = []  # split point is a (feature, value) pair
    # iterate over features
    for i in range(len(dataset[0].attributes)):
        value_label_pairs = []
        for item in dataset:
            value_label_pairs.append((item.attributes[i], item.label))
        sorted_value_label_pairs = sorted(value_label_pairs, key=lambda x: x[0])

        prev_label = sorted_value_label_pairs[0][1]
        prev_value = sorted_value_label_pairs[0][0]
        for (value, label) in sorted_value_label_pairs:
            if label != prev_label and value != prev_value:
                split_points.append((i, value))
            prev_label = label
            prev_value = value
    return split_points


def optimal_split_point(dataset, split_points):
    # Find split point with max IG
    dataset_entropy = set_entropy(dataset)
    dataset_size = len(dataset)
    IGs = []
    for (feature, value) in split_points:
        subset_l = []
        subset_r = []
        for item in dataset:
            if item.attributes[feature] < value:
                subset_l.append(item)
            else:
                subset_r.append(item)
        l_dataset_entropy = set_entropy(subset_l)
        r_dataset_entropy = set_entropy(subset_r)
        weighted_average_entropies = ((len(subset_l) / dataset_size) * l_dataset_entropy) \
                                     + ((len(subset_r) / dataset_size) * r_dataset_entropy)
        IG = dataset_entropy - weighted_average_entropies
        IGs.append(IG)
    return split_points[IGs.index(max(IGs))]


def split_dataset(dataset, feature, value):
    # Split dataset based on split point (feature, value)
    subset_l = []
    subset_r = []
    for item in dataset:
        if item.attributes[feature] < value:
            subset_l.append(item)
        else:
            subset_r.append(item)
    return subset_l, subset_r


def majority_label(dataset):
    return Counter([item.label for item in dataset]).most_common(1)[0][0]


def induce_decision_tree(dataset):
    split_points = potential_split_points(dataset)
    if not split_points:
        return leaf_node(majority_label(dataset))
    else:
        (optimal_feature, optimal_value) = optimal_split_point(dataset, split_points)
        node = split_node(dataset, optimal_feature, optimal_value)  # create node
        subset_l, subset_r = split_dataset(dataset, optimal_feature, optimal_value)  # split dataset

        l_child = induce_decision_tree(subset_l)  # create left child
        r_child = induce_decision_tree(subset_r)  # create right child

        node.add_left_child(l_child)
        node.add_right_child(r_child)

        return node


def traverse_tree(root, attributes):
    if not root:
        raise Exception
    if isinstance(root, leaf_node):
        return root.label
    else:
        feature = root.feature
        value = root.value
        if attributes[feature] < value:
            return traverse_tree(root.left, attributes)
        else:
            return traverse_tree(root.right, attributes)


def load_tree(file):
    output = open('./train_data/{}'.format(file), 'rb')
    tree = pickle.load(output)
    output.close()
    return tree


class DecisionTreeClassifier(object):
    """
    A decision tree classifier
    
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    
    """

    def __init__(self):
        self.is_trained = False
        self.tree = None

    def train(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the 
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array
        
        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
        # # check if train data already exists
        # if os.path.exists('./train_data/{}'.format(sys.argv[1])):
        #     # in this case we can use the stored data
        #     # and we dont have to recompute our tree
        #     self.tree = load_tree(sys.argv[1])
        #     self.is_trained = True
        #     return self

        data_set = [data_item(list(attrs) + [label]) for (attrs, label) in zip(x, y)]

        self.tree = induce_decision_tree(data_set)

        # # create dir if not exists
        # if not os.path.exists('./train_data'):
        #     os.makedirs('./train_data')
        #
        # # save tree
        # output = open('./train_data/{}'.format(sys.argv[1]), 'wb+')
        # pickle.dump(self.tree, output)
        # output.close()

        # output to console
        # print_node(self.tree, 0)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)
        
        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        try:
            # load decision tree from memory
            if not self.tree:
                self.tree = load_tree(sys.argv[1])
            predictions = np.array([traverse_tree(self.tree, test_instance) for test_instance in x])
        except Exception as e:
            print(e)
            return None

        # remember to change this if you rename the variable
        return predictions
