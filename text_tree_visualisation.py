from node import leaf_node, split_node
import math

feature_descriptions = {0: "x-box", 1: "y-box", 2: "width", 3: "high", 4: "onpix", 5: "x-bar", 6: "y-bar",
                        7: "x2bar", 8: "y2bar", 9: "xybar", 10: "x2ybr", 11: "xy2br", 12: "x-ege", 13: "xegvy",
                        14: "y-ege", 15: "yegvx"}


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


def print_leaf(leaf_node, depth):
    str = '{tabs}+---Leaf {label}'.format(tabs=add_tabs(depth), label=leaf_node.label)
    print(str)


def print_split(split_node, depth):
    current = '{tabs}+---Node {feature} < {value}, set entropy {entropy}'.format(
        tabs=add_tabs(depth), feature=feature_descriptions[split_node.feature], value=split_node.value,
        entropy=set_entropy(split_node.dataset))
    print(current)
    print_node(split_node.left, depth + 1)
    print_node(split_node.right, depth + 1)


def print_node(__obj__, depth=0):
    if isinstance(__obj__, leaf_node):
        return print_leaf(__obj__, depth)
    elif isinstance(__obj__, split_node):
        return print_split(__obj__, depth)
    else:
        return RuntimeError


def add_tabs(depth):
    str = ''
    for i in range(depth):
        str += '\t'
    return str
