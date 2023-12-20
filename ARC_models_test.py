import os
os.chdir('/Users/gustaw/Documents/ARC/simple_ARC')
from tqdm import tqdm 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle
from Models.model import VariationalAutoencoder

# from make_analogies_functions import *
# from helper_functions import *

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

# Load the model
encoder, decoder = VariationalAutoencoder(), VariationalAutoencoder() 
encoder.load_state_dict(torch.load('Models/encoder.pt'))
decoder.load_state_dict(torch.load('Models/decoder.pt'))

