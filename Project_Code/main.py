from Genomic_Preprocessing import load_vcf
from RBM_Model import RBM
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
mse = np.mean((data - recalled_data)**2)
print(f'Recall Error (MSE): {mse}')

# add noise to the data and apply error correction
corruption_level = 0.3
corrupted_data = rbm.add_noise(data, corruption_level)
corrected_data = rbm.error_correction(corrupted_data)
error_rate = np.mean((data != corrected_data).astype(int))
print(f'Error after Correction: {error_rate}')