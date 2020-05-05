import lightgbm as lgb
import numpy as np
import os

# ------------------------------------------------------------ GRADIENT BOOSTING

def load_booster(filename):

    loaded_model = lgb.Booster(model_file=filename)

    return loaded_model

def predict_booster(model, xs, best_iteration=True):
    
    if best_iteration:
        probs = model.predict(xs, num_iteration=model.best_iteration)
    else:
        probs = model.predict(xs)

    return np.argmax(probs, 1)

def save_model(model, save_dir):
    # save best iteration
    model.save_model(os.path.join(save_dir, "model.txt"))

def train_booster(train_xs, train_ys, val_xs, val_ys, params):

    lgb_train = lgb.Dataset(train_xs, train_ys)
    lgb_valid = lgb.Dataset(val_xs, val_ys)

    gbm = lgb.train(params, lgb_train, valid_sets=[lgb_valid])

    return gbm