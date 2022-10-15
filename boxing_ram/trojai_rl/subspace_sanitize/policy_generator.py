import numpy as np

'''
    a class to extract the policy function from the model architechture
'''
class PolicyGenerator:
    
    def __init__(self, model):
        self.model = model
        
    def policy(self, obs):
        dist, value = self.model(obs)
        return dist

'''
    a class that returns a sanitized policy function given the model architecture and the projection operator
'''
class SanitizedPolicyGenerator:
    
    def __init__(self, model, projection_operator, num_samples=None, sample_range=None):
        self.model = model
        self.projection_operator = projection_operator
        self.num_samples = num_samples
        self.sample_range = sample_range

    def policy(self, obs):
        obs = obs*255         # convert observation from floats to integer
        sanitized_obs = np.matmul(self.projection_operator, obs.T)
        sanitized_obs = sanitized_obs/255    # convert observation from integer to floats
        dist, value = self.model(sanitized_obs.T)
        return dist
    