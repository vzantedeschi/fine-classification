import numpy as np

from tqdm import tqdm

import torch

from src.monitors import MonitorTree

def train_stochastic(dataloader, model, optimizer, criterion, epoch, pruning=True, reg=1, norm=float("inf"), monitor=None):

    model.train()

    last_iter = epoch * len(dataloader)

    train_obj = 0.
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):

        optimizer.zero_grad()

        pred = model(batch["radiances"], batch["properties"])

        loss = criterion(pred, batch["test_properties"])

        if pruning:

            obj = loss + reg * torch.norm(model.sparseMAP.eta, p=norm)
            train_obj += obj.detach().numpy()

            pbar.set_description("avg train loss + reg %f" % (train_obj / (i + 1)))

        else:

            obj = loss
            train_obj += obj.detach().numpy()

            pbar.set_description("avg train loss %f" % (train_obj / (i + 1)))

        obj.backward()

        optimizer.step()

        if monitor:
            monitor.write(model, i + last_iter, check_pruning=False, train={"Loss": loss.detach()})

def evaluate(dataloader, model, criterion, epoch=None, monitor=None, classify=False):

    model.eval()

    total_loss = 0.
    predictions = []
    properties = []
    
    for i, batch in enumerate(dataloader):

        pred = model(batch["radiances"], batch["properties"])

        loss = criterion(pred, batch["test_properties"])
        total_loss += loss.detach()

        if classify:
            predictions.append(model.classify(batch["properties"]))
            properties.append(torch.cat((batch["properties"], batch["test_properties"]), 1).detach().numpy())

    if monitor:
        monitor.write(model, epoch, val={"Loss": total_loss})

    if classify:
        return total_loss.numpy() / len(dataloader), np.hstack(predictions), np.vstack(properties)
    else:
        return total_loss.numpy() / len(dataloader)