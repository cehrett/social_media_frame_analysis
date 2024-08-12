import os
import numpy as np
import pandas as pd
import csv
import torch

import matplotlib
import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt

import pyro
from pyro import poutine
from pyro.infer import MCMC, NUTS, HMC
import pyro.distributions as dist
import argparse

# Local code
from .utils.bayesian_account_clustering import fc_analysis_tools as at


# Settings, perhaps not user-configurable
# dtype for count data
dtype = torch.float32
burn_in = 200
num_samples_inference = 400
matplotlib.rcParams['lines.linestyle']=''


def run_analysis(cluster_label_loc, 
                 flags_loc,  
                 output_dir, 
                 flags, 
                 user_id='user_id',
                 post_id='post_id',
                 clusters_to_remove=[-1],
                 users_to_remove=[],
                 author_id_loc=None,
                 use_cuda=True, 
                 device_num=0):
    device = torch.device(f'cuda:{device_num}') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    if author_id_loc is None:
        author_id_loc = flags_loc

    # Load and preprocess data
    df_cluster_labels, df_flags_for_each_author, df_author_id_for_each_post = at.load_data(
        cluster_label_loc, 
        flags_loc, 
        author_id_loc, 
        flags, 
        user_id=user_id, 
        clusters_to_remove=clusters_to_remove, 
        users_to_remove=users_to_remove
    )
    df_model, df_successes = at.preprocess_data(df_author_id_for_each_post=df_author_id_for_each_post,
                                                df_author_id_for_each_flag=df_flags_for_each_author, 
                                                df_cluster_labels=df_cluster_labels, 
                                                flags=flags, 
                                                user_id=user_id, 
                                                post_id=post_id)
    df_authors = df_author_id_for_each_post[[post_id,user_id]].groupby([user_id], as_index=False).size().\
    rename(columns={'size':'trials'}).merge(df_flags_for_each_author, how='inner', on=[user_id])

    # Prepare data for the model
    data = at.prepare_model_data(df_model, flags, dtype, device)
    
    # Setup and run model
    lambda_loc, sigma_prior = at.setup_model(data, len(flags), device)
    def model(data):
        # global log-odds
        lam = pyro.sample("lambda", lambda_loc)
        sigma = pyro.sample("sigma", sigma_prior)
        with pyro.plate("cluster_labels", size = data['trials'].shape[0], device=device):
            # sample log odds for each player separately
            lam_n = pyro.sample("lambda_n", dist.Normal(lam, sigma).to_event(1))

            # observe the number of accounts on each cluster_label
            pyro.sample("flags", dist.Binomial(total_count=data['trials'], logits=lam_n).to_event(1), obs=data['successes'])
    samples = at.run_mcmc(model, data, burn_in, num_samples_inference)
    
    at.plot_lambdas(samples, data, device, flags, output_dir)

    # Analyze and save results
    df_kld, df_mean_shift = at.calculate_kld_mean_shift(samples, df_model, flags, device)  # Define this function
    df_suspicion = at.generate_suspicion_dataframe(df_kld, df_mean_shift, df_cluster_labels)
    top_cluster_labels = at.identify_top_clusters(df_suspicion)
    top_cluster_labels.to_csv(os.path.join(output_dir,'top_suspicious_frame_clusters.csv'), index=False, quoting=csv.QUOTE_ALL)
    df_successes.to_csv(os.path.join(output_dir,'frame_cluster_use_per_author.csv'), index=False)
    df_authors.to_csv(os.path.join(output_dir,'flags_and_num_posts_per_author.csv'), index=False)
    

if __name__ == "__main__":
    # Default paths and settings
    default_cluster_label_loc = os.path.join('/', 'home', 'cehrett', 'Projects', 'Trolls', 'narr_extraction', 'user_facing_repo', 'frame_cluster_results.csv')
    default_flags_loc = os.path.join('/', 'home', 'cehrett', 'Projects', 'Trolls', 'narr_extraction', 'user_facing_repo', 'data', 'sample_data_for_frame_extraction.csv')
    default_output_dir = os.path.join('/', 'home', 'cehrett', 'Projects', 'Trolls', 'narr_extraction', 'user_facing_repo', 'example_output')
    default_author_id_loc = default_flags_loc
    default_flags = ['in_US']
    default_min_num_trials = 5
    default_cuda = True
    default_device_num = 0
    
    parser = argparse.ArgumentParser(description='Analysis Script for Suspicious Cluster Identification')
    parser.add_argument('--cluster_label_loc', default=default_cluster_label_loc, help='Path to csv with cluster label for each post. Output of `cluster_frames.py`.')
    parser.add_argument('--flags_loc', default=default_flags_loc, help='Path to csv with flags for each author')
    parser.add_argument('--author_id_loc', default=default_flags_loc, help='Path to csv with author_id for each post')
    parser.add_argument('--output_dir', default=default_output_dir, help='Output directory to store results')
    parser.add_argument('--flags', default=default_flags, nargs='+', help='Flags of suspicious accounts included in data.')
    parser.add_argument('--min_num_trials', type=int, default=default_min_num_trials, help='Minimum number of trials')
    parser.add_argument('--cuda', action='store_true', default=default_cuda, help='Use CUDA if available')
    parser.add_argument('--device_num', type=int, default=default_device_num, help='CUDA device number')
    
    args = parser.parse_args()
    run_analysis(
        cluster_label_loc=args.cluster_label_loc,
        flags_loc=args.flags_loc,
        author_id_loc=args.author_id_loc,
        output_dir=args.output_dir,
        flags=args.flags,
        use_cuda=args.cuda,
        device_num=args.device_num
    )

    