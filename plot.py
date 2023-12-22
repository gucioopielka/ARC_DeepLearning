import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from itertools import combinations
import matplotlib.cm as cm


def plot_RDM_concept(rdm, analogy_set, title='RDM', save=False):

    analogy, analogy_idxs, analogy_detailed_idxs = analogy_set
    
    # Plot the RDM
    plt.imshow(1-rdm, cmap='coolwarm')
    plt.title(title, fontsize=16)

    # Analogies local
    for i in range(len(analogy_detailed_idxs) - 1):
        start = analogy_detailed_idxs[i]
        end = analogy_detailed_idxs[i + 1] - 1  # -1 because the end index is exclusive
        width = height = end - start

        # Draw a red rectangle
        rect = patches.Rectangle((start, start), 
                                width, 
                                height, 
                                linewidth=1, 
                                edgecolor='darkblue', 
                                facecolor='none')
        plt.gca().add_patch(rect)

    # Analogies global
    for i in range(len(analogy_idxs) - 1):
        start = analogy_idxs[i]
        end = analogy_idxs[i + 1] - 1  # -1 because the end index is exclusive
        width = height = end - start

        # Draw a red rectangle
        rect = patches.Rectangle((start, start), 
                                width, 
                                height, 
                                linewidth=1, 
                                edgecolor='purple', 
                                facecolor='none')
        plt.gca().add_patch(rect)

    midpoints = [(analogy_idxs[i] + analogy_idxs[i + 1]) / 2 for i in range(len(analogy))]

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Similarity', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.xticks(midpoints)
    plt.yticks(midpoints)

    plt.gca().set_xticklabels(analogy, rotation=90, fontsize=10)
    plt.gca().set_yticklabels(analogy, fontsize=10)

    plt.tight_layout()
    if save:
        plt.savefig(f'data/plots/{title}', bbox_inches='tight', dpi=300)
    if not save:
        plt.show()
    plt.close()

def get_rule_sim_diagonal(rdm, analogy_set):

    analogy, analogy_idxs, analogy_detailed_idxs = analogy_set

    rule_similarity = []
    for i in range(len(analogy_idxs) - 1):
        rule_similarity.append(np.mean(rdm[analogy_idxs[i]:analogy_idxs[i+1], analogy_idxs[i]:analogy_idxs[i+1]]))

    return pd.DataFrame({'Concept' : analogy, 'Similarity' : rule_similarity}).sort_values(by='Similarity', ascending=False)

def get_rule_sim_off_diagonal(rdm, analogy_set):

    analogy, analogy_idxs, analogy_detailed_idxs = analogy_set
    
    # Relate concepts to their RDM indices
    concept_indices = []
    for i in range(len(analogy_idxs) - 1):
        concept_indices.append((analogy[i] , (analogy_idxs[i], analogy_idxs[i+1])))

    # Generate all possible concept/indices pairs
    concept_pairs = []
    for pair in list(combinations(concept_indices, 2)):
        concept_dict = {
            'concept_1': pair[0][0],
            'indices_1': pair[0][1],
            'concept_2': pair[1][0],
            'indices_2': pair[1][1]
        }
        concept_pairs.append(concept_dict)

    # Compute the mean similarity for each concept pair
    pairs_df = pd.DataFrame({'Concept' : np.zeros(len(concept_pairs)), 'Similarity' : np.zeros(len(concept_pairs))})
    for pair_idx, pair in enumerate(concept_pairs):
        mean = np.mean(rdm[pair['indices_1'][0]:pair['indices_1'][1], pair['indices_2'][0]:pair['indices_2'][1]])
        pairs_df.loc[pair_idx, 'Concept'] = f'{pair["concept_1"]} - {pair["concept_2"]}'
        pairs_df.loc[pair_idx, 'Similarity'] = mean
    
    return pairs_df.sort_values(by='Similarity', ascending=False)

def plot_rule_similarity(df, title='RDM', save=False):
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10, 6))
    norm = plt.Normalize(-1, 1)
    colors = cm.coolwarm(norm(df['Similarity']*2.8))
    plt.barh(df['Concept'], df['Similarity'], color=colors)
    plt.xlabel('Average Pairwise D Similarity')
    plt.title(title)
    plt.gca().invert_yaxis()  # To display the highest value at the top
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.tick_params(axis='y', which='both', left=False)
    if save:
        plt.savefig(f'data/plots/{title}', bbox_inches='tight', dpi=300)
    if not save:
        plt.show()
    plt.close()


