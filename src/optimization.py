import numpy as np

from tqdm import tqdm

import torch
from torch.autograd import Variable

from src.trees import BinarySearchTree

# ----------------------------------------------------------------------- LOGISTIC REGRESSION

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size)     

    def forward(self, x):

        y_pred = torch.sigmoid(self.linear(x))
        
        return y_pred

# ----------------------------------------------------------------------- LATENT TREES REGRESSION

class LPSparseMAP(torch.nn.Module):

    def __init__(self, bst, dim):

        super(LPSparseMAP, self).__init__()

        self.A = torch.nn.Parameter(torch.rand(bst.nb_split, dim))
        self.bst = bst       

    def compute_q(self, x):

        # compute tree paths q
        XA = torch.mm(x, self.A.T)

        q = torch.ones((len(x), self.bst.nb_nodes))

        q[:, self.bst.desc_left] = torch.min(q[:, self.bst.split_nodes], XA[:, self.bst.split_nodes])
        q[:, self.bst.desc_right] = torch.min(q[:, self.bst.split_nodes], -XA[:, self.bst.split_nodes])

        return q

    # def gradient(self):

    #     q = self.compute_q(X)

    #     g = np.zeros(q.shape)

    #     # value when q > 1
    #     idx = q > 1
    #     g[idx] = 1

    #     # value when 0 <= q <= 1
    #     idx = 0 <= q <= 1
    #     g[idx] = q[idx]

    #     return g

    # def forward(self, X):

    #     q = self.compute_q(X)

    #     # differentiable output
    #     out = np.zeros(q.shape)

    #     # value when q > 1
    #     idx = q > 1
    #     out[idx] = q[idx] - 0.5

    #     # value when 0 <= q <= 1
    #     idx = 0 <= q <= 1
    #     out[idx] = q[idx]**2 / 2 

    #     return out

    def forward(self, X):

        q = self.compute_q(X)

        # non differentiable output
        z = torch.clamp(q, 0, 1)

        return z

def train(x, Y, nb_iter=1e4, lr=2e-1):

    n, d = x.shape

    # init latent tree
    bst = BinarySearchTree()

    # init latent tree optimizer
    sparseMAP = LPSparseMAP(bst, d)

    # init predictor ( [x;z]-> y )
    predictor = LogisticRegression(d+bst.nb_nodes, 1)

    # init optimizer
    optimizer = torch.optim.SGD(list(sparseMAP.parameters()) + list(predictor.parameters()), lr=lr)

    # init loss
    criterion = torch.nn.BCELoss(reduction="mean")

    t_y = torch.from_numpy(Y[:, None]).float()
    t_x = torch.from_numpy(x).float()

    predictor.train()
    sparseMAP.train()

    pbar = tqdm(range(int(nb_iter)))
    for i in pbar:

        optimizer.zero_grad()

        z = sparseMAP(t_x)

        xz = torch.cat((t_x, z), 1)

        y_pred = predictor(xz)   

        loss = criterion(y_pred, t_y)

        loss.backward()
        
        optimizer.step()

        pbar.set_description("BCE train loss %s" % loss.detach().data)

    return sparseMAP, predictor