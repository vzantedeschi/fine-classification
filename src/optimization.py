import numpy as np

from tqdm import tqdm

import torch

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

    def forward(self, x):

        q = self._compute_q(x)

        # non differentiable output for q > 1
        z = torch.clamp(q, 0, 1)

        return z

    def _compute_q(self, x):

        # compute tree paths q
        XA = torch.mm(x, self.A.T)

        q = torch.ones((len(x), self.bst.nb_nodes))

        q[:, self.bst.desc_left] = torch.min(q[:, self.bst.split_nodes], XA[:, self.bst.split_nodes])
        q[:, self.bst.desc_right] = torch.min(q[:, self.bst.split_nodes], -XA[:, self.bst.split_nodes])

        return q

    # def gradient(self):

    #     q = self._compute_q(X)

    #     g = np.zeros(q.shape)

    #     # value when q > 1
    #     idx = q > 1
    #     g[idx] = 1

    #     # value when 0 <= q <= 1
    #     idx = 0 <= q <= 1
    #     g[idx] = q[idx]

    #     return g

    # def objective(self, X):

    #     q = self._compute_q(X)

    #     # differentiable output
    #     out = np.zeros(q.shape)

    #     # value when q > 1
    #     idx = q > 1
    #     out[idx] = q[idx] - 0.5

    #     # value when 0 <= q <= 1
    #     idx = 0 <= q <= 1
    #     out[idx] = q[idx]**2 / 2 

    #     return out

class BinaryClassifier(torch.nn.Module):

    def __init__(self, bst, dim):

        super(BinaryClassifier, self).__init__()

        # init latent tree optimizer (x -> z)
        self.sparseMAP = LPSparseMAP(bst, dim)

        # init predictor ( [x;z]-> y )
        self.predictor = LogisticRegression(dim+bst.nb_nodes, 1)

    def forward(self, x):
        
        z = self.sparseMAP(x)

        xz = torch.cat((x, z), 1)

        return self.predictor(xz)

    def parameters(self):
        return list(self.sparseMAP.parameters()) + list(self.predictor.parameters())

    def predict(self, x):

        y_pred = self.forward(x)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return y_pred.detach()

    def train(self):
        self.sparseMAP.train()
        self.predictor.train()

def train(x, y, nb_iter=1e4, lr=5e-1):

    n, d = x.shape

    # init latent tree
    bst = BinarySearchTree()

    model = BinaryClassifier(bst, d)

    # init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # init loss
    criterion = torch.nn.BCELoss(reduction="mean")

    # cast to pytorch Tensors
    t_y = torch.from_numpy(y[:, None]).float()
    t_x = torch.from_numpy(x).float()

    model.train()

    pbar = tqdm(range(int(nb_iter)))
    for i in pbar:

        optimizer.zero_grad()

        y_pred = model(t_x)

        loss = criterion(y_pred, t_y)

        loss.backward()
        
        optimizer.step()

        pbar.set_description("BCE train loss %s" % loss.detach().numpy())

    return model