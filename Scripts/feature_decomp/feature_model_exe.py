import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE
# from umap import UMAP


for task in ['cue','stimuli']:

    X = np.load(f'../../data/model_X_{task}.npy')
    Y = np.load(f'../../data/model_Y_{task}.npy')

    method='t-SNE' # t-SNE feature decomposition
    transform = TSNE(n_components=2, n_iter=1000, random_state=42, perplexity=30)

    # method='UMAP'
    # transform = UMAP(n_neighbors=5, min_dist=0.3, n_components=2)
        
    X_2d = transform.fit_transform(X)

    plt.figure(figsize=(6, 6))
    unique_labels = np.unique(Y)
    
    if task =='cue': # choose color
        colors = sns.color_palette("husl", 4)
    if task =='stimuli':
        tab20_cmap = plt.get_cmap('tab20')
        colors = tab20_cmap.colors[:16]

    for label in unique_labels:
        label = int(label)
        indices = (Y == label)
        if task=='cue':
            plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c=colors[label], label=f'Class {label}', s=1)
        else:
            plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c=[colors[label% len(colors)]], label=f'Class {label}', s=1)
    
    plt.xlabel(f'Dimension 1 ({method})')
    plt.ylabel(f'Dimension 2 ({method})')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{method}',fontsize=17)
    plt.axis('off')
    
    plt.savefig(f'../../results/model_feature_{task}.png',dpi=500)
    plt.close()
