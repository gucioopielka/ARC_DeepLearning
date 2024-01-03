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

def one_hot_encoder(X,
                    n_channels=10, 
                    target_channel=1):
    
    if target_channel > 9:
        raise ValueError("target_channel must be between 0 and 9")

    # Create a one-hot encoded matrix
    one_hot = np.zeros((n_channels,) + X.shape, dtype=int)

    # Channel 0 encodes where X is 0
    one_hot[0, X == 0] = 1

    if target_channel != 0:
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
        
def preprocess_simpleARC(X_full, 
                         n_channels=10,
                         target_channel=1):

    X_full = X_full / 255

    X_full_mat = []
    for i in range(len(X_full)):
        X_sca = scaling(X_full[i], 30, 30)
        X_one = one_hot_encoder(X_sca, 
                                n_channels=n_channels, 
                                target_channel=target_channel)
        X_full_mat.append(X_one)

    return np.stack(X_full_mat)

def sort_analogies(tasks, 
                   indices):
    '''Sort the indices and tasks according to indices alphabetically'''
    tasks = [i for i in tasks] # convert to list
    paired_sorted = sorted(zip(indices, tasks), key=lambda pair: pair[0])
    analogy_idx, tasks = zip(*paired_sorted)
    analogy_idx, tasks= np.array(analogy_idx), np.array(tasks)

    print('Analogy counts:')
    analogy_idx_unique = np.unique(analogy_idx, return_counts=True)
    for i in range(len(analogy_idx_unique[0])):
        print(f'{analogy_idx_unique[0][i]} --- {analogy_idx_unique[1][i]}')

    return tasks, analogy_idx

def encode_analogy(analogy_idx):
    '''Encode the (sorted or unsorted) analogy index into integers'''
    unique_strings = np.unique(analogy_idx)
    string_to_int = {string: i for i, string in enumerate(unique_strings)}
    encoded_array = np.array([string_to_int[item] for item in analogy_idx])
    return encoded_array,


class SimpleARC_Dataset(Dataset):
    def __init__(self, inputs, outputs, indices=None):
        assert inputs.shape[0] == outputs.shape[0]
        self.inputs = inputs
        self.outputs = outputs
        self.indices = indices

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        if self.indices is not None:
            return self.inputs[idx], self.outputs[idx], self.indices[idx]
        else:
            return self.inputs[idx], self.outputs[idx]
    
def get_data_loader(tasks, 
                    target_channel,
                    batch_size, 
                    indices=None, 
                    shuffle=False,
                    n_channels=10):

    # Rescaling and one-hot encoding 
    # (n, 6, 10, 10) -> (n, 6, 10, 30, 30)
    tasks = np.stack([preprocess_simpleARC(task, n_channels=n_channels, target_channel=target_channel) for task in tasks]) 

    # Splitting into inputs and outputs and unrolling to (n*3, 10, 30, 30)
    n = len(tasks)
    inputs = tasks[:, :3, :, :].reshape((n*3, 10, 30, 30)) # First 3 are inputs
    outputs = tasks[:, 3:, :, :].reshape((n*3, 10, 30, 30)) # Last 3 are outputs
    if indices is not None:
        indices = np.repeat(indices, 3)

    inputs = torch.from_numpy(inputs).float()
    outputs = torch.from_numpy(outputs).float()
    if indices is not None:
        indices = torch.from_numpy(indices).float() 

    dataset = SimpleARC_Dataset(inputs, outputs, indices)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def pairwise_cosine_sim(vec):
    # Normalize to unit vectors
    epsilon = 1e-8  # Add a small value to avoid division by zero
    norms = vec.norm(p=2, dim=1, keepdim=True) + epsilon
    vec_normalized = vec / norms
    # Similarity
    sim_matrix = torch.mm(vec_normalized, vec_normalized.t())
    # Get the upper triangle without the diagonal (k=1)
    sim_values = sim_matrix[torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)]
    # Average cosine distance
    return 1 - sim_values.mean()

def calculate_d_penalty(mu, rule_ids):
    mu_inp, mu_out = mu.chunk(2, dim=0)  # Split into input and output
    unique_rules = rule_ids.unique()  # Get unique rules
    d_penalties = [] # Initialize list to store D penalties for each rule

    for rule in unique_rules:
        # Get the indices for the current rule
        current_rule_indices = (rule_ids == rule).nonzero(as_tuple=True)[0]

        # Extract mu for the current rule using the indices
        rule_mu_inp = mu_inp[current_rule_indices]
        rule_mu_out = mu_out[current_rule_indices]

        # Calculate the D's as differences of mu (mean representation in latent space)
        if current_rule_indices.numel() > 1: 
            diff = rule_mu_out - rule_mu_inp
            d_penalties.append(pairwise_cosine_sim(diff))

    return torch.stack(d_penalties).nanmean()


def accuracy(X_inp, X_out, exclude_zero=False):
    per_diff = []

    # Option to exclude the color zero for accuarcy calculations
    if exclude_zero:
        for i in range(len(X_inp)):
            raw_diff = np.count_nonzero(np.logical_and(X_inp[i] == X_out[i], X_inp[i] != 0))
            (per_diff.append(raw_diff / np.count_nonzero(X_inp[i])) if np.count_nonzero(X_inp[i]) != 0 else per_diff.append(0))

    else:
        for i in range(len(X_inp)):
            raw_diff = np.count_nonzero(X_inp[i] == X_out[i])
            per_diff.append(raw_diff / X_inp[i].size)

    return per_diff

def validate_d_penalty(model, valid_loader):
    model.eval()
    d_pos = []
    with torch.no_grad():
        for batch_idx, (input, output, rule_ids) in enumerate(valid_loader):
            in_out = torch.cat((input, output), dim=0).to(device)
            out, mu, logVar = model(in_out)
            d_pos.append(calculate_d_penalty(mu, rule_ids))
    
    return torch.stack(d_pos).nanmean()

def validate_reconstruction(model, valid_loader):
    # Put model in evaluation mode and start reconstructions based on latent vector
    model.eval()
    X_inp, X_out = [], []
    with torch.no_grad():
        for batch_idx, (input, output, rule_ids) in enumerate(valid_loader):
            in_out = torch.cat((input, output), dim=0).to(device)
            out, mu, logVar = model(in_out)
            for i in range(len(in_out)):
                X_inp.append(reverse_one_hot_encoder(in_out[i].cpu().numpy()))
                X_out.append(reverse_one_hot_encoder(out[i].cpu().numpy()))
    
    acc_inc_zero = np.mean(accuracy(X_inp, X_out, exclude_zero=False))
    acc_non_zero = np.mean(accuracy(X_inp, X_out, exclude_zero=True))

    return acc_inc_zero, acc_non_zero

def get_latent_vec(encoder, data_loader):
    encoder.eval()

    with torch.no_grad():
        diff_vec = []
        for batch_idx, (input, output) in tqdm(enumerate(data_loader), total=len(data_loader)):
            in_out = torch.cat((input, output), dim=0).to(device)
            mu, logVar = encoder.encode(in_out)
            mu_inp, mu_out = mu.chunk(2, dim=0)  # Split into input and output
            diff = mu_out - mu_inp
            diff_vec.append(diff.cpu().detach().numpy())

    return np.concatenate(diff_vec, axis=0)

def get_RDM(mat):
    mat_flattened = np.reshape(mat, (mat.shape[0], -1))
    return pairwise_distances(mat_flattened, metric='cosine')