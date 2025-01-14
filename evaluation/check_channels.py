import os 
from pathlib import Path
current_file_path = os.path.dirname(os.path.realpath(__file__))
project_dir = Path(current_file_path).parent
os.chdir(project_dir)

from tqdm import tqdm 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
from sklearn.model_selection import train_test_split

from evaluation.plot import *
from training.model import VariationalAutoencoder
from make_analogies.helper_functions import *


batch_size = 2500
n_channels = 10


### PREPARE DATA
with open("data/analogy_11000_tasks.pkl", "rb") as f:
    tasks = pickle.load(file=f)
with open("data/analogy_11000_index_global.pkl", "rb") as f:
    global_index = pickle.load(file=f)
with open("data/analogy_11000_index_local.pkl", "rb") as f:
    local_index = pickle.load(file=f)

# Split into train and validation sets (same as training)
_, tasks, _, global_index, _, local_index = train_test_split(
    tasks, global_index, local_index, test_size=0.3, random_state=69
    )

# Sort the local_index and tasks according to local_index, alphabetically
tasks, local_index = sort_analogies(tasks, local_index)

# Prepare the analogy names for plotting
analogy_set = preprocess_analogy_names(global_index, local_index) 


### LOAD MODEL
encoder = torch.load('data/models/encoder_fullARC.pth', map_location=device).eval()


### RUN 
for target_channel in range(10):
    print(f'Channel {target_channel}')

    valid_loader = get_data_loader(tasks, 
                                target_channel=target_channel, 
                                n_channels=n_channels, 
                                batch_size=batch_size, 
                                shuffle=False)
    
    diff_vec = get_latent_vec(encoder, valid_loader)
    rdm = get_RDM(diff_vec)


    ### PLOT

    plot_RDM_concept(rdm, analogy_set, title=f'RDM Channel {target_channel}', save=True)
    
    df_rule_diag = get_rule_sim_diagonal(1-rdm, analogy_set)
    plot_rule_similarity(df_rule_diag, title=f'Diagonal Similarity Channel {target_channel}', save=True)
    
    df_rule_off_diag = get_rule_sim_off_diagonal(1-rdm, analogy_set)
    df_rule_off_diag_top5= pd.concat((df_rule_off_diag[:5], df_rule_off_diag[-5:]))
    plot_rule_similarity(df_rule_off_diag_top5, title=f'Off-diagonal Similarity Channel {target_channel}', save=True)

