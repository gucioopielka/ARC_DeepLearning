import os 
from pathlib import Path
current_file_path = os.path.dirname(os.path.realpath(__file__))
project_dir = Path(current_file_path).parent
os.chdir(project_dir)

import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from make_analogies.helper_functions import *
from training.model import *

n_channels = 10
target_channel = 1
batch_size = 2500
task_optimizer_level = 'global'


with open("data/analogy_11000_tasks.pkl", "rb") as f:
    tasks = pickle.load(file=f)
with open("data/analogy_11000_index_global.pkl", "rb") as f:
    global_analogy_index = pickle.load(file=f)
with open("data/analogy_11000_index_local.pkl", "rb") as f:
    local_analogy_index = pickle.load(file=f)

# Get task/analogy indices
indices = eval(f'{task_optimizer_level}_analogy_index')
encoded_indices, encoding_mapping = encode_analogy(indices)

# Split into train and validation sets
tasks_train, tasks_valid, analogy_idx_train, analogy_idx_valid = train_test_split(
    tasks, encoded_indices, test_size=0.3, random_state=69
    )
train_loader = get_data_loader(tasks_train, 
                               indices=analogy_idx_train, 
                               batch_size=batch_size,
                               n_channels=n_channels, 
                               target_channel=target_channel, 
                               shuffle=False)
valid_loader = get_data_loader(tasks_valid, 
                               indices=analogy_idx_valid, 
                               batch_size=batch_size, 
                               n_channels=n_channels, 
                               target_channel=target_channel, 
                               shuffle=False)

# Train encoder
encoder = VariationalAutoencoder(img_channels=n_channels).to(device)
encoder = train_encoder(encoder, 
                        train_loader, 
                        valid_loader, 
                        epochs=20, 
                        rule_lambda=1e6, 
                        lr=1e-3, 
                        loss_fn='reconstruction + d', 
                        device=device)


# Map validation analogy indices to names and save for later model evaluation
analogy_idx_valid_named = [analogy_name for encoded_idx in analogy_idx_valid for analogy_name, encoding in encoding_mapping.items() if encoding == encoded_idx]

save_dict = {
    'state_dict': encoder.state_dict(), 
    'analogy_idx': analogy_idx_valid_named
}
torch.save(save_dict, f'data/models/encoder_{n_channels}_{1}_{batch_size}.pth')





