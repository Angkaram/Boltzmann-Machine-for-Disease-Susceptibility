from RBM_Model import RBM

def extract_features(rbm, data):
    # extract the hidden layer activations as features
    h_prob, h_sample = rbm.sample_hidden(data.T)
    
    # return the hidden layer samples as features
    return h_sample.T