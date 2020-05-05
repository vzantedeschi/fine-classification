import numpy as np

class BinarySearchTree():

    def __init__(self, depth=2):

        # 2d data, decision tree of depth D=2
        # max num nodes T = 2^(D+1) - 1 = 7 nodes.
        #
        #      0
        #    /   \
        #   1     2
        #  / \   / \
        # 3   4 5   6

        self.nb_nodes = 2**(depth+1) - 1
        self.nb_split = 2**depth - 1

        self.split_nodes = range(self.nb_split)

        self.desc_left = range(1, self.nb_nodes, 2)
        self.desc_right = range(2, self.nb_nodes, 2)