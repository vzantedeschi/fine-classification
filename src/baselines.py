import torch

from sklearn.cluster import AgglomerativeClustering

# ----------------------------------------------------------------------- BASELINES - REGRESSION

class LinearRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(LinearRegression, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size, bias=False)     

    def forward(self, x):
        
        return self.linear(x)

    def predict(self, x):

        return self.forward(x).detach()

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size, bias=False)     

    def forward(self, x):

        y_pred = torch.sigmoid(self.linear(x))
        
        return y_pred

    def predict(self, x):

        return self.forward(x).detach()

class NonLinearRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(NonLinearRegression, self).__init__()

        self.network = []
        self.network.append(torch.nn.Linear(in_size, in_size, bias=True))
        self.network.append(torch.nn.ReLU())
        self.network.append(torch.nn.Linear(in_size, in_size, bias=True))
        self.network.append(torch.nn.ReLU())
        self.network.append(torch.nn.Linear(in_size, out_size, bias=True))

        self.network = torch.nn.Sequential(*self.network)

    def forward(self, x1, x2):
        
        return self.network(torch.cat((x1, x2), 1))

    def predict(self, x1, x2):

        return self.forward(x1, x2).detach()


class AgglomerativeRegressor(torch.nn.Module):

    def __init__(self, nb_clusters, in_size1, in_size2, out_size):

        super(LinearRegressor, self).__init__()

        # init clustering (x2 -> z)
        self.nb_clusters = nb_clusters
        self.clustering = AgglomerativeClustering(n_clusters=nb_clusters)

        # init predictor ( [x1;z]-> y )
        self.predictor = LinearRegression(in_size1 + 1 + nb_clusters, out_size)

    def eval(self):
        self.predictor.eval()

    def forward(self, X1, X2):
        
        # add offset
        x1 = torch.cat((X1, torch.ones((len(X1), 1))), 1)
        x2 = X2.numpy()

        z = torch.from_numpy(AgglomerativeClustering.fit_predict(x2)[:, None])

        xz = torch.cat((x1, z), 1)

        return self.predictor(xz)

    def parameters(self):
        return list(self.predictor.parameters())

    def predict(self, X1, X2):

        y_pred = self.forward(X1, X2)

        return y_pred.detach()

    def classify(self, X2):

        x2 = X2.numpy()

        return AgglomerativeClustering.fit_predict(x2)

    def train(self):
        self.predictor.train()