import os
from tqdm import tqdm 
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
from Models.model import VariationalAutoencoder
from make_analogies.helper_functions import preprocess_simpleARC
from sklearn.metrics import pairwise_distances

batch_size = 1024

with open("data/nonduplicates.pkl", "rb") as f:
    tasks = pickle.load(file=f)
with open("data/analogy_index.pkl", "rb") as f:
    analogy_index = pickle.load(file=f)
with open("data/analogy_index_detailed.pkl", "rb") as f:
    analogy_index_detailed = pickle.load(file=f)

# Sort the analogy_index_detailed and tasks according to analogy_index_detailed
tasks = [i for i in tasks] # convert to list
paired_sorted = sorted(zip(analogy_index_detailed, tasks), key=lambda pair: pair[0])
analogy_idx, tasks = zip(*paired_sorted)
analogy_idx, tasks= np.array(analogy_idx), np.array(tasks)
print('Analogy counts:')
analogy_idx_unique = np.unique(analogy_idx, return_counts=True)
for i in range(len(analogy_idx_unique[0])):
    print(f'{analogy_idx_unique[0][i]} --- {analogy_idx_unique[1][i]}')


def get_data_loader(tasks, target_channel, batch_size):

    n = len(tasks)

    # Rescaling and one-hot encoding 
    # (n, 6, 10, 10) -> (n, 6, 10, 30, 30)
    target_channel = 2 # Channel to one-hot encode
    tasks = np.stack([preprocess_simpleARC(task, target_channel=target_channel) for task in tasks]) 

    # Splitting into inputs and outputs and unrolling to (n*3, 10, 30, 30)
    inputs = tasks[:, :3, :, :].reshape((n*3, 10, 30, 30)) # First 3 are inputs
    outputs = tasks[:, 3:, :, :].reshape((n*3, 10, 30, 30)) # Last 3 are outputs

    inputs = torch.from_numpy(inputs).float()
    outputs = torch.from_numpy(outputs).float()
    dataset = torch.utils.data.TensorDataset(inputs, outputs)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_latent_vec(encoder, data_loader):

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



# Load the model
encoder = torch.load('Models/encoder.pt', map_location=device).eval()
decoder = torch.load('Models/decoder.pt', map_location=device).eval()

# Run
for channel in range(1, 10):
    print(f'Channel {channel}')
    data_loader = get_data_loader(tasks, target_channel=channel, batch_size=batch_size)
    diff_vec = get_latent_vec(encoder, data_loader)
    rdm = get_RDM(diff_vec)
    np.save(f'data/rdms/channel_{channel}.npy', rdm)
