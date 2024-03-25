# Code in this script was adapted from code originally written by Hudson Smith.
# for details, see: https://arxiv.org/abs/2401.06205
import torch
import pyro.distributions as dist
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_prior_membership_probability_samples(components, sig, n_samp = 10000):
    component_probs = torch.tensor(components)
    component_probs = component_probs / component_probs.sum()
    mu = torch.log(component_probs)
    sig = torch.tensor(sig)
    
    ell_dist = dist.Normal(mu, sig)
    ell_samples = ell_dist.sample((n_samp,))
    return torch.softmax(ell_samples, -1)

def compute_prior_membership_probability_intervals(samples, q = (0.025, 0.975)):
    p_quantiles = torch.quantile(samples, q = torch.tensor(q), axis=0)
    
    df = pd.DataFrame(p_quantiles.cpu().numpy().T, columns = q)
    
    return df

def plot_prior_membership_distributions(samples):
    for k, s in enumerate(samples.T):
        sns.kdeplot(s.cpu().numpy(), label=f"k={k}", log_scale=(False, False))
        
    plt.legend()

