class split_node(object):
    def __init__(self, dataset, feature, value):
        self.dataset = dataset
        self.feature = feature
        self.value = value
        self.left = None
        self.right = None

    def add_left_child(self, left_child):
        self.left = left_child

    def add_right_child(self, right_child):
        self.right = right_child


class leaf_node(object):
    def __init__(self, label):
        self.label = label
