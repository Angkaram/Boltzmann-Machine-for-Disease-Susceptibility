import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_features(features):
    # use t-SNE to reduce the dimensionality of the features to 2D
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    
    # create a scatter plot of the reduced features
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    plt.title('t-SNE Visualization of RBM Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
