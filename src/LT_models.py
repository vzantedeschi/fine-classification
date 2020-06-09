import numpy as np
import torch

from src.trees import BinarySearchTree

# ----------------------------------------------------------------------- LATENT TREES REGRESSION

class LPSparseMAP(torch.nn.Module):

    def __init__(self, bst_depth, dim, pruned=True):

        super(LPSparseMAP, self).__init__()

        self.bst = BinarySearchTree(bst_depth)       
        self.A = torch.nn.Parameter(torch.rand(self.bst.nb_split, dim))
        
        self.pruned = pruned
        if pruned:
            self.eta = torch.nn.Parameter(torch.rand(self.bst.nb_nodes))

    def forward(self, x):

        q = self._compute_q(x)
        
        if self.pruned:

            self.d = self._compute_d(q)
            z = torch.clamp(q, 0, 1)
            z = torch.min(z, self.d)

        else:
            z = torch.clamp(q, 0, 1)

        return z

    def predict(self, x):

        z = self.forward(x).detach().numpy()

        return self.bst.predict(z)

    def _compute_d(self, q):
        
        # init d and colors different for all nodes
        d = self.eta.clone()
        coloring = np.arange(self.bst.nb_nodes)

        for c in coloring:
            d[c] = self._compute_d_colored(q, [c]) 

        while True:
            max_violating_d = - np.inf
            max_violating_ix = None

            for t in range(1, self.bst.nb_nodes):
                # if edge is violating, and is larger than max so far
                p = self.bst.parent(t)
                if d[t] > d[p] and d[t] > max_violating_d:
                    max_violating_d = d[t]
                    max_violating_ix = t

            if max_violating_ix is None:
                # no more violations, we are done
                break

            # fix the selected violating edge, propagating along color.
            # invariant: always keep the color of the parent.
            p = self.bst.parent(max_violating_ix)
            pc = coloring[p]
            coloring[coloring == max_violating_ix] = pc

            pc_ix = (coloring == pc)
            d[pc_ix] = self._compute_d_colored(q, pc_ix)

        d = torch.clamp(d, 0, 1)
        
        return d

    def _compute_d_colored(self, q, idx):

        topk = 0
        nb_k = 0

        d = torch.mean(self.eta[idx])

        # select qs greater than current d (violating the constraints)
        q_sorted, _ = torch.sort(q[:, idx][q[:, idx] >= d], descending=True)

        for k in range(len(q_sorted)):
            if d > q_sorted[k]:
                break
                
            topk += q_sorted[k]
            nb_k += 1

            d = (torch.sum(self.eta[idx]) + topk) / (len(self.eta[idx]) + nb_k)
        
        return d

    def _compute_q(self, x):

        # compute tree paths q
        XA = torch.mm(x, self.A.T)

        q = torch.ones((len(x), self.bst.nb_nodes))

        # upper bound children's q to parent's q        
        # trick to avoid inplace operations involving A
        q[:, self.bst.desc_left] = torch.min(q[:, self.bst.split_nodes], XA[:, self.bst.split_nodes])
        q[:, self.bst.desc_right] = torch.min(q[:, self.bst.split_nodes], -XA[:, self.bst.split_nodes])

        for _ in range(self.bst.depth):
            q[:, self.bst.desc_left] = torch.min(q[:, self.bst.desc_left], q[:, self.bst.split_nodes])
            q[:, self.bst.desc_right] = torch.min(q[:, self.bst.desc_right], q[:, self.bst.split_nodes])

        return q

class BinaryClassifier(torch.nn.Module):

    def __init__(self, bst_depth, dim, pruned=True):

        super(BinaryClassifier, self).__init__()

        # init latent tree optimizer (x -> z)
        self.sparseMAP = LPSparseMAP(bst_depth, dim, pruned)

        # init predictor ( [x;z]-> y )
        self.predictor = LogisticRegression(dim + self.sparseMAP.bst.nb_nodes, 1)

    def eval(self):
        self.sparseMAP.eval()
        self.predictor.eval()

    def forward(self, X):
        
        # add offset
        x = torch.cat((X, torch.ones((len(X), 1))), 1)

        z = self.sparseMAP(x)

        xz = torch.cat((x, z), 1)

        return self.predictor(xz)

    def parameters(self):
        return list(self.sparseMAP.parameters()) + list(self.predictor.parameters())

    def predict(self, X):

        y_pred = self.forward(X)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return y_pred.detach()

    def classify(self, X):

        # add offset
        x = torch.cat((X, torch.ones((len(X), 1))), 1)

        return self.sparseMAP.predict(x)

    def train(self):
        self.sparseMAP.train()
        self.predictor.train()

class LinearRegressor(torch.nn.Module):

    def __init__(self, bst_depth, in_size1, in_size2, out_size, pruned=True):

        super(LinearRegressor, self).__init__()

        # init latent tree optimizer (x2 -> z)
        self.sparseMAP = LPSparseMAP(bst_depth, in_size2 + 1, pruned)

        # init predictor ( [x1;z]-> y )
        self.predictor = LinearRegression(in_size1 + 1 + self.sparseMAP.bst.nb_nodes, out_size)

    def eval(self):
        self.sparseMAP.eval()
        self.predictor.eval()

    def forward(self, X1, X2):
        
        # add offset
        x1 = torch.cat((X1, torch.ones((len(X1), 1))), 1)
        x2 = torch.cat((X2, torch.ones((len(X2), 1))), 1)

        z = self.sparseMAP(x2)

        xz = torch.cat((x1, z), 1)

        return self.predictor(xz)

    def parameters(self):
        return list(self.sparseMAP.parameters()) + list(self.predictor.parameters())

    def predict(self, X1, X2):

        y_pred = self.forward(X1, X2)

        return y_pred.detach()

    def classify(self, X2):

        x2 = torch.cat((X2, torch.ones((len(X2), 1))), 1)

        return self.sparseMAP.predict(x2)

    def train(self):
        self.sparseMAP.train()
        self.predictor.train()