import numpy as np
import pickle
import sys
import text_tree_visualisation

from node import leaf_node, split_node
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

MAX_DEPTH = 10

codes = [
    Path.MOVETO,
    Path.LINETO,
]


def no_descendants(node):
    if isinstance(node, split_node):
        return 2 + no_descendants(node.left) + no_descendants(node.right)
    else:
        return 0


def create_vert(x_parent, y_parent, x_child, y_child):
    return [(x_parent, y_parent), (x_child, y_child)]


class Visualiser(object):

    def __init__(self, name=None, tree_root=None, max_depth=MAX_DEPTH):
        if not tree_root:
            output = open('./train_data/{}'.format(name), 'rb')
            self.tree_root = pickle.load(output)
            output.close()
        else:
            self.set_tree(tree_root)
        self.MAX_DEPTH = max_depth
        fig, self.ax = plt.subplots()

    def set_tree(self, tree_root):
        self.tree_root = tree_root

    def draw(self):
        coord_list = self.get_coords_of_tree(0, self.tree_root, 0, True)
        m = np.transpose(coord_list)

        self.ax.plot(m[0], m[1], 'ro')
        plt.show()

    def draw_text(self):
        text_tree_visualisation.print_node(self.tree_root)

    # get coordinates of child node
    def get_coord(self, x_parent, child, depth=0, is_left_child=False):
        if depth == 0:
            x, y = 0, 0
        else:
            y = -depth

            grandchild = None
            x = 1
            if isinstance(child, split_node):
                x += 1
                if is_left_child:
                    grandchild = child.right
                else:
                    grandchild = child.left

            x += no_descendants(grandchild)

            if is_left_child:
                x = -x
            x += x_parent

            # create edges on graph
            vert = create_vert(x_parent, 1-depth, x, y)

            path = Path(vert, codes)

            patch = patches.PathPatch(path, lw=2)
            self.ax.add_patch(patch)

        # add text
        if isinstance(child, leaf_node):
            label = child.label
        else:
            label = "x{} < {}".format(child.feature, child.value)
        self.ax.text(x, y, label)

        return [[x, y]]

    def get_coords_of_tree(self, x_parent, subtree, depth, is_left):
        subtree_coords = self.get_coord(x_parent, subtree, depth=depth, is_left_child=is_left)
        if depth >= self.MAX_DEPTH:
            return subtree_coords
        if isinstance(subtree, leaf_node):
            return subtree_coords
        if isinstance(subtree, split_node):
            return subtree_coords + self.get_coords_of_tree(
                subtree_coords[0][0], subtree.left, depth + 1, True) + self.get_coords_of_tree(
                subtree_coords[0][0], subtree.right, depth + 1, False)


if __name__ == "__main__":
    v = Visualiser(name=sys.argv[1], max_depth=50)
    v.draw()
