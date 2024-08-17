from Genomic_Preprocessing import load_vcf
from RBM_Model import RBM
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from features import extract_features
from visual import visualize_features
from advanced_visual import visualize_weight_matrix, visualize_feature_space, compare_reconstructions

file_path = 'HG00096.cnvnator.illumina_high_coverage.20190825.sites.vcf.gz'

# load and preprocess data
data = load_vcf(file_path)
print(data.shape)

# initialize RBM with visible and hidden units
visible_units = data.shape[1]
hidden_units = 64
rbm = RBM(visible_units, hidden_units)

# train RBM on the preprocessed data
rbm.train(data, epochs=10, lr=0.1)

# evaluate the RBM by calculating the recall error
recalled_data = rbm.recall(data)
mse = np.mean((data - recalled_data) ** 2)
print(f'Recall Error (MSE): {mse}')

# add noise to the data and apply error correction
corruption_level = 0.3
corrupted_data = rbm.add_noise(data, corruption_level)
corrected_data = rbm.error_correction(corrupted_data)
error_rate = np.mean((data != corrected_data).astype(int))
print(f'Error after Correction: {error_rate}')

# new for week 2:

features = extract_features(rbm, data)

# visualize the extracted features
# visualize_features(features) ## this is done below with the new advanced visual

# new for Week 3:

# 1. visualize the weight matrix
visualize_weight_matrix(rbm)

# 2. visualize the feature space using PCA
visualize_feature_space(features, method='pca')

# 3. visualize the feature space using t-SNE
visualize_feature_space(features, method='tsne')

# 4. compare the original and reconstructed data (WORK IN PROGRESS, needs reshaping?)
#reconstructed_data = rbm.sample_visible(rbm.sample_hidden(data)[1])[1].T
#compare_reconstructions(data, reconstructed_data)
