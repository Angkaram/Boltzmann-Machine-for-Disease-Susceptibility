import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_weight_matrix(rbm):
    """
    visualizes the weight matrix of the rbm model.
    :param rbm: trained rbm model, which includes the weight matrix to be visualized.
    """
    plt.figure(figsize=(10, 8))
    # image
    plt.imshow(rbm.weights, cmap='viridis', aspect='auto')
    
    # the color bar shows the magnitude of weights, with positive and negative values represented by different colors
    cbar = plt.colorbar()
    cbar.set_label('Weight Magnitude') 
    
    # set the title of the plot
    plt.title("Weight Matrix Visualization")
    
    plt.xlabel("Visible Units")
    plt.ylabel("Hidden Units")
    
    plt.show()

def visualize_feature_space(features, labels=None, method='pca'):
    """
    visualizes the feature space using either pca or t-sne for dimensionality reduction.
    :param features: data features extracted from the rbm, which will be visualized.
    :param labels: optional, the labels for the data points, used for coloring the scatter plot.
    :param method: the dimensionality reduction technique to use, 'pca' for principal component analysis or 'tsne' for t-distributed stochastic neighbor embedding.
    """
    if method == 'pca':
        # initialize pca to reduce dimensions to 2
        reducer = PCA(n_components=2)
        title = "Feature Space Visualization (PCA)"
    elif method == 'tsne':
        # initialize t-sne to reduce dimensions to 2 with specified parameters
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
        title = "Feature Space Visualization (t-SNE)"
    else:
        # raise an error if the method is not recognized
        raise ValueError("Method must be 'pca' or 'tsne'")

    # apply the dimensionality reduction technique to the features
    reduced_features = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    
    if labels is not None: # in my code, this will likely be None, but I am adding this in case I expand later
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
        plt.colorbar()
    else:
        # scatter plot without labels
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    
    plt.title(title)
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.show()

def compare_reconstructions(original, reconstructed):
    """
    compares the original and reconstructed data samples.
    :param original: original data samples, typically images or feature vectors.
    :param reconstructed: data samples reconstructed by the rbm, which are compared with the originals.
    """
    # a figure with a 2x10 grid of subplots
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    
    # iterate through the first 10 samples for comparison
    for i in range(10):
        # original data sample in the top row
        axes[0, i].imshow(original[i].reshape((28, 28)), cmap='gray')
        axes[0, i].axis('off') 
        axes[0, i].set_title("Original")
        
        # the reconstructed data sample in the bottom row
        axes[1, i].imshow(reconstructed[i].reshape((28, 28)), cmap='gray')
        axes[1, i].axis('off') 
        axes[1, i].set_title("Reconstructed")

    plt.suptitle("Original vs Reconstructed Data")
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.show()
