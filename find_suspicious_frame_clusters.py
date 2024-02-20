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


# Settings, perhaps not user-configurable
# dtype for count data
dtype = torch.float32
burn_in = 200
num_samples_inference = 400
matplotlib.rcParams['lines.linestyle']=''


# Helper functions       
def get_prior_log_probs(samples):
    # compute sample log-probs from the prior (not hyperprior)
    lam = samples['lambda'][:,None]
    sig = samples['sigma'][:,None]
    lamn = samples['lambda_n']
    
    return dist.Normal(lam, sig).log_prob(lamn)

def get_posterior_log_probs(samples):
    # model the posterior as a normal distribution to get sample log probabilities
    lamn = samples['lambda_n']
    loc = lamn.mean(axis=0)
    scale = lamn.std(axis=0)
    
    return dist.Normal(loc, scale).log_prob(lamn)

def kld(logp, logq):
    # expectation of log(p/q) is just the mean for our samples
    # note: by this sampling definition, kld is not garunteed to be positive
    # it will almost always be so as long as the samples are correctly drawn from p. 
    return torch.mean(logp - logq, axis=0)

def mean_shift(samples):
    lam = samples['lambda'][:,None]
    lamn = samples['lambda_n']
    
    return lamn.mean(axis=0) - lam.mean(axis=0)

def get_ex_frame(row, df_cluster_frames):
    # Given a row of df_suspicion, can provide an example frame
    # Filter the df_cluster_frames to get rows with matching 'cluster_labels'
    matching_theories = df_cluster_frames[df_cluster_frames['cluster_labels'] == row['cluster_labels']]['frames']
    
    # Randomly sample a theory if there are any matching
    if not matching_theories.empty:
        return np.random.choice(matching_theories)
    else:
        return None
    
def row_with_max_kld(group):
    # Returns the row with the max 'kld' value for each group
    return group.loc[group['kld'].idxmax()]
        
        
def load_data(cluster_label_loc, flags_loc, author_id_loc, flags):
    df_cluster_labels = pd.read_csv(cluster_label_loc)
    df_cluster_labels = df_cluster_labels[[col for col in df_cluster_labels.columns if col != 'embeddings']]
    
    df_flags_for_each_author = pd.read_csv(flags_loc)
    df_author_id_for_each_post = pd.read_csv(author_id_loc)  
    
    # Deduplicate and remove unneeded cols
    df_flags_for_each_author = df_flags_for_each_author[['author_id'] + flags].groupby('author_id').agg(any).reset_index() 
    
    return df_cluster_labels, df_flags_for_each_author, df_author_id_for_each_post


def preprocess_data(df_author_id_for_each_post, df_cluster_labels, flags):
    # import pdb; pdb.set_trace()
    merged_df = pd.merge(df_author_id_for_each_post, df_cluster_labels, on='id')
    df_successes = merged_df.groupby(['author_id', 'cluster_labels']).size().reset_index(name='successes')

    flag_set = set(flags) 
    df_authors_long = df_author_id_for_each_post.melt(id_vars=[col for col in df_author_id_for_each_post.columns if col not in flags], var_name='flag', value_name='has_flag').query('flag in @flag_set')

    df_model = df_successes.merge(df_authors_long, how='left', on=['author_id']).groupby(['cluster_labels', 'flag']).agg({'has_flag': 'sum', 'author_id': 'nunique'}).rename(columns={'author_id': 'num_accounts'}).reset_index().pivot_table(index=['cluster_labels', 'num_accounts'], columns='flag', values='has_flag').reset_index()
    
    # Convert flag columns to numeric
    for flag in flags:
        df_model[flag] = pd.to_numeric(df_model[flag], errors='coerce').astype(int)
    # import pdb; pdb.set_trace()

    return df_model, df_successes


def setup_model(data, num_flags, device):
    # Define hyperpriors
    lambda_loc = dist.Normal(-2.*torch.ones(num_flags).to(device), scale=1.).to_event(1)
    sigma_prior = dist.HalfNormal(1.0*torch.ones(num_flags).to(device)).to_event(1)
    
    # Setup model - This could include pyro.render_model() if needed
    return lambda_loc, sigma_prior


def run_mcmc(model, data, burn_in, num_samples_inference):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples_inference, warmup_steps=burn_in)
    mcmc.run(data)
    return mcmc.get_samples()


def plot_posterior_coeffs(lambda_samples, observed_rates, axs, label, device, eps=0):
    quantiles = torch.quantile(lambda_samples, q=torch.tensor([0.1, 0.5, 0.9]).to(device), dim=0).cpu().numpy()
    quantiles = 1/(1+np.exp(-quantiles))

    for ix, ax in enumerate(axs):
        # import pdb; pdb.set_trace
        obs = observed_rates[...,ix]
        med = quantiles[1,...,ix]
        yerr = np.abs(quantiles[[0,2],...,ix].transpose((1,0)) - med[...,None]).transpose((1,0))
        ax.errorbar(obs+eps, med, yerr=yerr , fmt="", errorevery=4, label=label)


def run_analysis(args):
    device = torch.device(f'cuda:{args.device_num}') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # Load and preprocess data
    df_cluster_labels, df_flags_for_each_author, df_author_id_for_each_post = load_data(args.cluster_label_loc, args.flags_loc, args.author_id_loc, args.flags)
    df_model, df_successes = preprocess_data(df_author_id_for_each_post, df_cluster_labels, args.flags)
    df_authors = df_author_id_for_each_post[['id','author_id']].groupby(['author_id'], as_index=False).size().\
    rename(columns={'size':'trials'}).merge(df_flags_for_each_author, how='inner', on=['author_id'])
    # import pdb; pdb.set_trace()

    # Prepare data for the model
    data = prepare_model_data(df_model, args.flags, dtype, device)
    
    # Setup and run model
    lambda_loc, sigma_prior = setup_model(data, len(args.flags), device)
    def model(data):
        # global log-odds
        lam = pyro.sample("lambda", lambda_loc)
        sigma = pyro.sample("sigma", sigma_prior)
        with pyro.plate("cluster_labels", size = data['trials'].shape[0], device=device):
            # sample log odds for each player separately
            lam_n = pyro.sample("lambda_n", dist.Normal(lam, sigma).to_event(1))

            # observe the number of accounts on each cluster_label
            pyro.sample("flags", dist.Binomial(total_count=data['trials'], logits=lam_n).to_event(1), obs=data['successes'])
    samples = run_mcmc(model, data, burn_in, num_samples_inference)
    
    plot_lambdas(samples, data, device, args.flags)

    # Analyze and save results
    df_kld, df_mean_shift = calculate_kld_mean_shift(samples, df_model, args.flags, device)  # Define this function
    df_suspicion = generate_suspicion_dataframe(df_kld, df_mean_shift, df_cluster_labels)
    top_cluster_labels = identify_top_clusters(df_suspicion)
    top_cluster_labels.to_csv(os.path.join(args.output_dir,'top_suspicious_frame_clusters.csv'), index=False, quoting=csv.QUOTE_ALL)
    df_successes.to_csv(os.path.join(args.output_dir,'frame_cluster_use_per_author.csv'), index=False)
    df_authors.to_csv(os.path.join(args.output_dir,'flags_and_num_posts_per_author.csv'), index=False)

    plt.show()  # Display plots
    

def plot_lambdas(samples, data, device, flags):
    # Process and visualize results
    fig, axs = plt.subplots(nrows=len(flags), sharex=True, sharey=True)
    fig.set_size_inches(8,12)

    axs = [axs]
    lambda_global_samples = samples['lambda'][:,None].repeat(1,data['trials'].shape[0], 1)
    lambda_n_samples = samples['lambda_n']
    observed_rates = calculate_observed_rates(data, device)  
    ref_line_x = np.linspace(0.001, 0.999, 100)
    # import pdb; pdb.set_trace()
    plot_posterior_coeffs(lambda_n_samples, observed_rates, axs, label='Part. P', device=device, eps=0.01) 
    plot_posterior_coeffs(lambda_global_samples, observed_rates, axs, label='Part. P Global', device=device) 
    for ix, ax in enumerate(axs):
        ax.plot(ref_line_x, ref_line_x, '--', label='p = obs. rate', color='k')
        ax.set_ylabel(f"Chance of success\n{flags[ix]}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid()
        
    axs[0].set_title(r"Posterior intervals for $\lambda$ coefficients")
    axs[-1].legend(loc=4)
    axs[-1].set_xlabel("Observed rate")
    plt.savefig('posterior_intervals_for_lambda_coefs.png')
    
    
def prepare_model_data(df_model, flags, dtype, device):
    # trials
    trials = torch.tensor(df_model.num_accounts.values, dtype=dtype).to(device)

    # successes 
    # import pdb; pdb.set_trace()
    successes = torch.tensor(df_model[flags].values, dtype=dtype).to(device)

    # put trials and successes into a single data structure for use by the models below
    data = dict(
        trials = trials[...,None],
        successes = successes,
    )
    
    return data

    
def calculate_observed_rates(data, device):
    """
    Calculate observed rates from data.
    """
    observed_rates = (data['successes'] / data['trials']).cpu().numpy()
    return observed_rates


def setup_plots(num_flags):
    """
    Initialize matplotlib plots.
    """
    fig, axs = plt.subplots(nrows=num_flags, sharex=True, sharey=True)
    fig.set_size_inches(8, 12)
    if num_flags == 1:
        axs = [axs]  # Ensure axs is a list for
    return fig, axs

def calculate_kld_mean_shift(samples, df_model, flags, device):
    """
    Calculate KLD and mean shift for the provided samples.
    """
    prior_logp = get_prior_log_probs(samples)
    posterior_logp = get_posterior_log_probs(samples)
    klds = kld(posterior_logp, prior_logp)
    mean_shifts = mean_shift(samples)

    df_kld = pd.DataFrame(klds.cpu().numpy(), index=df_model.cluster_labels, columns=flags).reset_index().melt(id_vars=['cluster_labels'], var_name='flag', value_name='kld')
    df_mean_shift = pd.DataFrame(mean_shifts.cpu().numpy(), index=df_model.cluster_labels, columns=flags).reset_index().melt(id_vars=['cluster_labels'], var_name='flag', value_name='mean_shift')

    return df_kld, df_mean_shift

def generate_suspicion_dataframe(df_kld, df_mean_shift, df_cluster_labels):
    """
    Generate a dataframe with suspicion metrics.
    """
    df_suspicion = pd.concat([df_kld, df_mean_shift.loc[:, 'mean_shift']], axis=1)
    df_suspicion['sampled_frame'] = df_suspicion.apply(lambda row: get_ex_frame(row, df_cluster_labels), axis=1)
    return df_suspicion

def identify_top_clusters(df_suspicion):
    """
    Identify top clusters based on KLD and mean shift.
    """
    top_cluster_labels = df_suspicion.query('mean_shift>0').groupby('cluster_labels').apply(row_with_max_kld)[['kld', 'flag', 'sampled_frame', 'cluster_labels']].sort_values('kld', ascending=False).reset_index(drop=True)
    return top_cluster_labels


# Default paths and settings
default_cluster_label_loc = os.path.join('/', 'home', 'cehrett', 'Projects', 'Trolls', 'narr_extraction', 'user_facing_repo', 'frame_cluster_results.csv')
default_flags_loc = os.path.join('/', 'home', 'cehrett', 'Projects', 'Trolls', 'narr_extraction', 'user_facing_repo', 'data', 'sample_data_for_frame_extraction.csv')
default_output_dir = os.path.join('/', 'home', 'cehrett', 'Projects', 'Trolls', 'narr_extraction', 'user_facing_repo', 'example_output')
default_author_id_loc = default_flags_loc
default_flags = ['in_US']
default_min_num_trials = 5
default_cuda = True
default_device_num = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Analysis Script for Suspicious Cluster Identification')
    parser.add_argument('--cluster_label_loc', default=default_cluster_label_loc, help='Path to csv with cluster label for each post')
    parser.add_argument('--flags_loc', default=default_flags_loc, help='Path to csv with flags for each author')
    parser.add_argument('--author_id_loc', default=default_flags_loc, help='Path to csv with author_id for each post')
    parser.add_argument('--output_dir', default=default_output_dir, help='Output directory to store results')
    parser.add_argument('--flags', default=default_flags, nargs='+', help='Flags of suspicious accounts included in data.')
    parser.add_argument('--min_num_trials', type=int, default=default_min_num_trials, help='Minimum number of trials')
    parser.add_argument('--cuda', action='store_true', default=default_cuda, help='Use CUDA if available')
    parser.add_argument('--device_num', type=int, default=default_device_num, help='CUDA device number')
    return parser.parse_args()


def main():
    args = parse_args()
    # import pdb; pdb.set_trace()
    run_analysis(args)
    

if __name__ == "__main__":
    main()