import os 
from pathlib import Path
current_file_path = os.path.dirname(os.path.realpath(__file__))
project_dir = Path(current_file_path).parent
os.chdir(project_dir)

import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW 
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from make_analogies.helper_functions import *
from training.model import VariationalAutoencoder

n_channels = 10


with open("data/nonduplicates.pkl", "rb") as f:
    tasks = pickle.load(file=f)
with open("data/analogy_index.pkl", "rb") as f:
    global_analogy_index = pickle.load(file=f)
with open("data/analogy_index_detailed.pkl", "rb") as f:
    local_analogy_index = pickle.load(file=f)

# Get task/analogy indices
task_optimizer_level = 'global'
indices = eval(f'{task_optimizer_level}_analogy_index')
encoded_indices = encode_analogy(indices)

# Split into train and validation sets
tasks_train, tasks_valid, analogy_idx_train, analogy_idx_valid = train_test_split(tasks, encoded_indices, test_size=0.3, random_state=69)
train_loader = get_data_loader(tasks_train, analogy_idx_train, batch_size=1024, n_channels=n_channels, target_channel=1, shuffle=False)
valid_loader = get_data_loader(tasks_valid, analogy_idx_valid, batch_size=1024, n_channels=n_channels, target_channel=1, shuffle=False)



# Initialize model
encoder = VariationalAutoencoder(img_channels=n_channels).to(device)



list(next(iter(train_loader))[2])

def train_encoder(model, 
          train_loader, 
          valid_loader,
          epochs=50, 
          rule_lambda=5e4, 
          lr=1e-3, 
          loss_fn = 'reconstruction + d'):
    optimizer = AdamW(model.parameters(), lr, weight_decay=0.2)
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (input, output, rule_ids) in enumerate(train_loader):

            # Combine input & output, adding noise, attaching to device
            in_out = torch.cat((input, output), dim=0)
            in_out = in_out.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = model(in_out)

            kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            bce = F.binary_cross_entropy(out, in_out, reduction='sum') 

            if loss_fn == 'reconstruction + d':
                d_positive, _ = calculate_d_penalty(mu, rule_ids)
                loss = bce + kl_divergence + rule_lambda*d_positive
            elif loss_fn == 'reconstruction':
                loss = bce + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        
        d_pos_train, d_neg_train = validate_d_penalty(model, 'training')
        print(f'Epoch {epoch+1}: Loss {round(bce.item() + kl_divergence.item(), 2)} - D {round(d_pos_train.mean().item(), 2)}')
        acc_zero_train, acc_non_zero_train = validate_reconstruction(model, train_loader)
        acc_zero_valid, acc_non_zero_valid, = validate_reconstruction(model, valid_loader)
        print('All Pixels : {0:.2f} ({2:.2f})% - Non-Zero Pixels : {1:.2f} ({3:.2f})%'.format(acc_zero_train*100, acc_non_zero_train*100, acc_zero_valid*100, acc_non_zero_valid*100))

    return model
