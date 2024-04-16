# Code in this script was adapted from code originally written by Hudson Smith.
# for details, see: https://arxiv.org/abs/2401.06205
import math
import torch
from torch import nn
from torch.distributions import constraints
import numpy as np
import pyro
import pyro.distributions as dist


class CoordinationModel:
    def __init__(self, 
                 data, 
                 device, 
                 alpha_encoder, 
                 alpha_logit_prior, 
                 beta_gamma_prior,
                 num_components,
                 num_flags, 
                 num_narratives,
                 supervised=False,
                 cholesky_rank=None):
        
        self.data = data
        self.device = device
        self.encoder = alpha_encoder
        self.alpha_logit_prior = alpha_logit_prior
        self.beta_gamma_prior = beta_gamma_prior
        self.num_components = num_components
        self.num_flags = num_flags
        self.num_narratives = num_narratives
        self.supervised = supervised
        
        self.latent_dim = num_flags+num_narratives
        self.init_scale = 1.0 # scale to use in covariance matrix
        self.rank = int(round(self.latent_dim**0.5)) if cholesky_rank is None else cholesky_rank
    
    def model(self, batch_ix):
        
        with pyro.plate('cluster', size=self.num_components):
            beta_gamma = pyro.sample('beta_gamma', self.beta_gamma_prior)
        
        if self.num_flags>0:
            # flag reg coeffs
            beta = beta_gamma[:,:self.num_flags]
            beta_p = torch.sigmoid(beta)

        if self.num_narratives>0:
            # narr reg coeffs
            gamma = beta_gamma[:,-self.num_narratives:]
            gamma_p = torch.sigmoid(gamma)

        with pyro.plate('author', size=self.data['trials'].shape[0], device=self.device, subsample=batch_ix):
            # log odds of membership in group
            alpha_logit = pyro.sample("alpha_logit", self.alpha_logit_prior)
            
            if self.supervised:
                alpha_p = pyro.sample('alpha', dist.OneHotCategorical(logits=alpha_logit), 
                                      obs=self.data['alphas_onehot'][batch_ix])
            else:
                # if we do not observe the group membership, then we marginalize over group assignments
                alpha_p = pyro.sample('alpha', dist.OneHotCategorical(logits=alpha_logit))

            if self.num_flags>0:
                # observe flags
                # alpha_p @ beta_p.T is marginalization over group assignments
                pyro.sample("flags", dist.Bernoulli(probs=(alpha_p @ beta_p).clip(0,1)).to_event(1), 
                            obs=self.data['flags'][batch_ix])

            if self.num_narratives>0:
                # observe narratives
                pyro.sample("narratives", dist.Binomial(total_count=self.data['trials'][batch_ix], 
                                                      probs=(alpha_p @ gamma_p).clip(0,1)).to_event(1), 
                            obs=self.data['successes'][batch_ix])
                
    def guide(self, batch_ix):
        
        with pyro.plate('cluster', self.num_components):
            beta_gamma = pyro.sample('beta_gamma', self.get_beta_gamma_posterior())
        
        feature_tensors = []
        if self.num_flags>0:
            feature_tensors.append(self.data['flags'][batch_ix])

        if self.num_narratives>0:
            feature_tensors.append(self.data['successes'][batch_ix])
            entropy = self.estimate_narrative_entropy(
                self.data['trials'][batch_ix], 
                self.data['successes'][batch_ix])
            feature_tensors.append(entropy[...,None])

        # concatenate feature tensors from flags and narratives
        encoder_features = torch.concat(feature_tensors, dim=-1)

        # alpha
        encoder = pyro.module("alpha_logit_encoder", self.encoder)
        alpha_logit_loc, _ = encoder(encoder_features)

        with pyro.plate('author', size=self.data['trials'].shape[0], device=self.device, subsample=batch_ix):
            alpha_logit = pyro.sample("alpha_logit", dist.Delta(alpha_logit_loc).to_event(1))
            
#     def get_beta_gamma_posterior(self, pars = None):
#         if pars is None:
#             ref_tensor = self.beta_gamma_prior.loc # reference tensor to help get dtype and device when initializing new tensors
#             #ref_tensor = self.beta_gamma_prior.base_dist.loc # reference tensor to help get dtype and device when initializing new tensors
#             scale = pyro.param('scale', ref_tensor.new_full((self.latent_dim,), 0.5**0.5 * self.init_scale), constraint = dist.constraints.softplus_positive)
#             loc = pyro.param('beta_gamma_loc', ref_tensor.new_empty(self.num_components, self.latent_dim).normal_(0, 1 / self.rank**0.5))
#             cov = pyro.param('beta_gamma_cov', ref_tensor.new_empty(self.num_components, self.latent_dim, self.rank).normal_(0, 1 / self.rank**0.5))
#         else:
#             scale = pars['scale']
#             loc = pars['beta_gamma_loc']
#             cov = pars['beta_gamma_cov']
        
#         return dist.LowRankMultivariateNormal(loc, cov*scale[...,None], scale**2)

    def get_beta_gamma_posterior(self, pars = None):
        if pars is None:
            ref_tensor = self.beta_gamma_prior.base_dist.loc # reference tensor to help get dtype and device when initializing new tensors
            loc = pyro.param('beta_gamma_loc', ref_tensor.new_empty(self.num_components, self.latent_dim).normal_(0, 1 / self.rank**0.5))
            scale = pyro.param('beta_gamma_scale', ref_tensor.new_full((self.num_components, self.latent_dim,), 0.5**0.5 * self.init_scale), 
                               constraint = dist.constraints.softplus_positive)
        else:
            loc = pars['beta_gamma_loc']
            scale = pars['beta_gamma_scale']
        
        return dist.Normal(loc, scale).to_event(1)
            
    # Get the log-odds point estimate for the accounts in the input dataset
    def compute_alpha_logodds(self, data):
        with torch.no_grad():
            feature_tensors = [data['flags'], data['successes']]
            if data['successes'].shape[-1]>0:
                feature_tensors.append(self.estimate_narrative_entropy(data['trials'], data['successes'])[...,None])
            alpha_logits, _ = self.encoder.eval()(
                torch.concat(feature_tensors, dim=-1)
            )
            
        return alpha_logits.cpu()
    
    @staticmethod
    def estimate_narrative_entropy(trials, successes, eps=1.e-8):
        rates = successes / (trials+1)
        entropy = -(rates * torch.log(rates+eps)).sum(axis=-1)
        
        return entropy
    
class AlphaNet(nn.Module):
    def __init__(self, num_flags, num_narratives, num_components, dropout_rate=0.5, use_entropy=True):
        super().__init__()
        num_hidden_1 = num_flags + num_narratives + num_flags * num_narratives
        num_hidden_2 = max(int(num_hidden_1//1.75), 5)
        num_hidden_3 = max(int(num_hidden_2//1.75), 5)
        
        num_input = num_flags+num_narratives+1
        if num_narratives==0:
            num_input -= 1

        self.enc_common = nn.Sequential(
            nn.Linear(num_input, num_hidden_1),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(num_hidden_1, affine=False),
            nn.Sigmoid(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.BatchNorm1d(num_hidden_2, affine=False),
            nn.Sigmoid(),
            nn.Linear(num_hidden_2,num_hidden_3),
            nn.BatchNorm1d(num_hidden_3, affine=False),
            nn.Sigmoid(),
        )
        
        self.loc = nn.Linear(num_hidden_3, num_components)
        self.log_scale = nn.Linear(num_hidden_3, num_components)
       
    def forward(self, x):
        h = self.enc_common(x)
        return self.loc(h), torch.exp(self.log_scale(h))