import numpy as np

class BinarySearchTree():

    def __init__(self, depth=2):

        # Example: decision tree of depth=2
        # max num nodes T = 2^(depth+1) - 1 = 7 nodes.
        #
        #      0
        #    /   \
        #   1     2
        #  / \   / \
        # 3   4 5   6

        self.depth = depth

        self.nb_nodes = 2**(depth+1) - 1
        self.nb_split = 2**depth - 1

        self.split_nodes = range(self.nb_split)
        self.leaves = range(self.nb_split, self.nb_nodes)

        self.desc_left = range(1, self.nb_nodes, 2)
        self.desc_right = range(2, self.nb_nodes, 2)

    def predict(self, z):
        """ each leaf corresponds to a class """

        labels = np.zeros(len(z))

        for i, leaf in enumerate(self.leaves):
            labels[z[:, leaf] == 1] = i

        return labels