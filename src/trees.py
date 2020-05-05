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

        self.des_left = list(range(1, self.nb_nodes, 2))
        self.des_right = list(range(2, self.nb_nodes, 2))

def get_q(X, A, b, bst):    

    XA = np.dot(X, A.T) # n_samples by n_split
    q = np.ones((len(X), bst.nb_nodes))

    q[:, bst.des_left] = np.minimum(q[:, bst.split_nodes], (XA[:, bst.split_nodes] - b[bst.split_nodes]))
    q[:, bst.des_right] = np.minimum(q[:, bst.split_nodes], (b[bst.split_nodes] - XA[:, bst.split_nodes]))

    return q

def forward(X):

    bst = BinarySearchTree()

    A = np.random.rand(bst.nb_split, X.shape[1])
    b = np.random.rand(bst.nb_split)

    q = get_q(X, A, b, bst)

    print(q)
