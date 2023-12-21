# Import standard library packages
import os
import json
import random
import warnings
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
warnings.filterwarnings("ignore") # to reduce unnecessary output
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Import external packages
import numpy as np

import torch
import torch.distributions
from scipy.stats import mode
#from stepwise_regression.step_reg import forward_regression, backward_regression
from torch.utils.data import Dataset, DataLoader
# Assign device used for script (e.g., model, processing)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

our_data_dir = '/Users/gustaw/Documents/ARC'


# Padding of the ARC matrices for convolutional processing
def padding(X, height=30, width=30, direction='norm'):
    h = X.shape[0]
    w = X.shape[1]

    a = (height - h) // 2
    aa = height - a - h

    b = (width - w) // 2
    bb = width - b - w

    if direction == 'norm':
        X_pad = np.pad(X, pad_width=((a, aa), (b, bb)), mode='constant')

    # Reverse padding for rescaling
    else:
        if height == 30 and width == 30:
            X_pad = X[:, :]
        elif height == 30:
            X_pad = X[:, abs(bb):b]
        elif width == 30:
            X_pad = X[abs(aa):a, :]
        else:
            X_pad = X[abs(aa):a, abs(bb):b]

    return X_pad

# Scaling of the ARC matrices using the Kronecker Product, retaining all the information
def scaling(X, height=30, width=30, direction='norm'):
    h = height/X.shape[0]
    w = width/X.shape[1]
    d = np.floor(min(h, w)).astype(int)

    X_scaled = np.kron(X, np.ones((d, d)))

    if direction == 'norm':
        return padding(X_scaled, height, width).astype(int)

    # Retain information for reverse scaling
    else:
        return d, X_scaled.shape

# Reverse scaling of the ARC matrices for final computations
def reverse_scaling(X_orig, X_pred):
    d, X_shape = scaling(X_orig, 30, 30, direction='rev') # get scaling information
    X_pad_rev = padding(X_pred, X_shape[0], X_shape[1], direction='rev') # reverse padding

    mm = X_shape[0] // d
    nn = X_shape[1] // d
    X_sca_rev = X_pad_rev[:mm*d, :nn*d].reshape(mm, d, nn, d)

    X_rev = np.zeros((mm, nn)).astype(int)
    for i in range(mm):
        for j in range(nn):
            X_rev[i,j] = mode(X_sca_rev[i,:,j,:], axis=None, keepdims=False)[0]

    return X_rev

def one_hot_encoder(X, target_channel=1):
    # Ensure target_channel is within the valid range and not 0
    if not 1 <= target_channel < 10:
        raise ValueError("target_channel must be between 1 and 9")

    # Create a one-hot encoded matrix
    one_hot = np.zeros((10,) + X.shape, dtype=int)

    # Channel 0 encodes where X is 0
    one_hot[0, X == 0] = 1

    # Target_channel encodes where X is 1
    one_hot[target_channel, X == 1] = 1

    return one_hot

# Reverse One-Hot-Encoding for easier visualization & final computations
def reverse_one_hot_encoder(X):
    return np.argmax(np.transpose(X, axes=[1,2,0]), axis=-1)

# Replace values in array with new ones from dictionary
def replace_values(X, dic):
    return np.array([dic.get(i, -1) for i in range(X.min(), X.max() + 1)])[X - X.min()]

# Convert ARC grids into numpy arrays
def get_all_matrix(X_full):
    X_fill = []
    for X_task in X_full:
        for X_single in X_task:
            X_fill.append(np.array(X_single))

    return X_fill

# Apply scaling, padding, and one-hot-encoding to arrays to get finalized grids
def get_final_matrix(X_full, stage="train"):
    if stage != "train":
        X_full = get_all_matrix(X_full)

    X_full_mat = []
    for i in range(len(X_full)):
        X_sca = scaling(X_full[i], 30, 30)
        X_one = one_hot_encoder(X_sca)
        X_full_mat.append(X_one)

    return X_full_mat

# Augment color of grids: randomly assign new colors to each color within a grid (creates 9 copies of original)
def augment_color(X_full, y_full):
    X_flip = []
    y_flip = []
    for X, y in zip(X_full, y_full):
        X_rep = np.tile(X, (10, 1, 1))
        X_flip.append(X_rep[0])
        y_rep = np.tile(y, (10, 1, 1))
        y_flip.append(y_rep[0])
        for i in range(1, len(X_rep)):
            rep = np.arange(10)
            orig = np.arange(10)
            np.random.shuffle(rep)
            dic = dict(zip(orig, rep))
            X_flip.append(replace_values(X_rep[i], dic))
            y_flip.append(replace_values(y_rep[i], dic))

    return X_flip, y_flip

# Augment orientation of grids: randomly rotates certain grids by 90, 180, or 270 degrees
def augment_rotate(X_full, y_full):
    X_rot = []
    y_rot = []
    for X, y in zip(X_full, y_full):
        k = random.randint(0, 4)
        X_rot.append(np.rot90(X, k))
        y_rot.append(np.rot90(y, k))

    return X_rot, y_rot

# Midpoint mirroring of grids: Creates copies of grids and mirrors at midpoint the left side
def augment_mirror(X_full, y_full):
    X_mir = []
    y_mir = []
    for X, y in zip(X_full, y_full):
        X_mir.append(X)
        y_mir.append(y)

        X_rep = X.copy()
        n = X_rep.shape[1]
        for i in range(n // 2):
            X_rep[:, n - i - 1] = X_rep[:, i]

        y_rep = y.copy()
        n = y_rep.shape[1]
        for i in range(n // 2):
            y_rep[:, n - i - 1] = y_rep[:, i]

        X_mir.append(X_rep)
        y_mir.append(y_rep)

    return X_mir, y_mir

# Combines array creation, augmentation, and preprocessing (e.g., scaling)
def preprocess_matrix(X_full, y_full, aug=[True, True, True]):
    X_full = get_all_matrix(X_full)
    y_full = get_all_matrix(y_full)

    if aug[0]:
        print("Augmentation: Random Color Flipping")
        X_full, y_full = augment_color(X_full, y_full)

    if aug[1]:
        print("Augmentation: Random Rotation")
        X_full, y_full = augment_rotate(X_full, y_full)

    if aug[2]:
        print("Augmentation: Midpoint Mirroring")
        X_full, y_full = augment_mirror(X_full, y_full)

    X_full = get_final_matrix(X_full)
    y_full = get_final_matrix(y_full)

    return X_full, y_full

# Defining loaders function to ease data preparation
def data_load(X_train, y_train, inp_index=None, stage="train", aug=[False, False, False], batch_size=1, shuffle=False):
    # Define what augmentations are applied, batch size, and if shuffling is desired

    data_set = ARCDataset(X_train, y_train, inp_index, stage=stage, aug=aug)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    del data_set

    return data_loader

def parse_arc_items(task_files, dataset_path):
    # Load items from file list and open JSON files
    focus_tasks = []
    for task_file in task_files:
        file = os.path.join(dataset_path, task_file)
        with open(file, 'r') as f:
            task = json.load(f)
            focus_tasks.append(task)

    # train: train (example) inputs (X) and output (y)
    # test: test inputs (X) and output (y)
    X_test, y_test, X_train, y_train = [[] for _ in range(4)]

    # Distinguish between train (example) and test grids (input vs output)
    for task in focus_tasks:
        Xs_test, ys_test, Xs_train, ys_train = [[] for _ in range(4)]

        for pair in task["test"]:
            Xs_test.append(pair["input"])
            ys_test.append(pair["output"])

        for pair in task["train"]:
            Xs_train.append(pair["input"])
            ys_train.append(pair["output"])

        X_test.append(Xs_test)
        y_test.append(ys_test)
        X_train.append(Xs_train)
        y_train.append(ys_train)
    
    return X_test, y_test, X_train, y_train

# PyTorch Dataset Framework: Custom processing of data (incl. Augmentations, Padding)
class ARCDataset(Dataset):
    def __init__(self, X, y, inp_index=None, stage="train", aug=[True, True, True]):
        self.stage = stage
        if inp_index is None:
            self.inp_index = np.insert(np.cumsum([len(i) for i in X]), 0, 0)
        else:
            self.inp_index = np.cumsum(np.insert(np.diff(inp_index), 0, 0)) # Making sure the index start with 0

        if self.stage == "train":
            self.X, self.y = preprocess_matrix(X, y, aug)
        else:
            self.X = get_final_matrix(X, self.stage)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.inp_index is not None:
            rule_id = next((i for i, boundary in enumerate(self.inp_index) if idx < boundary), len(self.inp_index)) - 1
        else:
            rule_id = []
        inp = self.X[idx]
        inp = torch.tensor(inp, dtype=torch.float32)

        if self.stage == "train":
            outp = self.y[idx]
            outp = torch.tensor(outp, dtype=torch.float32)
            return inp, outp, rule_id
        else:
            return inp, rule_id
        
def preprocess_simpleARC(X_full, target_channel=1):

    X_full = X_full / 255

    X_full_mat = []
    for i in range(len(X_full)):
        X_sca = scaling(X_full[i], 30, 30)
        X_one = one_hot_encoder(X_sca, target_channel=target_channel)
        X_full_mat.append(X_one)

    return np.stack(X_full_mat)
        